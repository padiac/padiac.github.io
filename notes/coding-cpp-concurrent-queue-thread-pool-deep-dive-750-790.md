## 1. 並發隊列的基礎演進 (750-760)

### 1.1 從“報錯”到“等待” (750 vs 760)

* **750 基礎型**:
    ```cpp
    // 邏輯: 不等待, 直接拋異常
    void push(T value) {
        std::lock_guard<std::mutex> lck(mut);
        if (que.size() >= max) throw concurrent_queue_full();
        que.push(value);
    }
    ```
* **760 混合阻塞型 (`concurrent_queue_cv`)**:
    ```cpp
    // Pop 完全阻塞, Push 輪詢 (混和模式)
    void push(T value) {
        std::unique_lock<std::mutex> uniq_lck(mut);
        while (que.size() > max) { // 輪詢
            uniq_lck.unlock();
            std::this_thread::sleep_for(50ms);
            uniq_lck.lock();
        }
        que.push(value);
        cond_var.notify_one();
    }
    
    void pop(T& value) {
        std::unique_lock<std::mutex> uniq_lck(mut);
        cond_var.wait(uniq_lck, [this] {return !que.empty();}); // 阻塞等待
        value = que.front(); que.pop();
    }
    ```

### 1.2 條件變量 (CV) 的精確使用

* **虛假喚醒 (Spurious Wakeup)**: 使用 Predicate Lambda 防止線程意外醒來.
* **通知策略**: 使用 `notify_one()` 減少線程競爭 (驚群效應).

---

## 2. 線程池：單隊列 vs 多隊列 (770-785)

### 2.1 770：全域共享隊列 (Lock Contention)

* 所有 Worker 共用同一個隊列, 爭搶一把鎖. 適用於任務較重的場景.

### 2.2 780：分而治之 (Multiple Queues per Thread)

* **內存管理**: 爲什麼用 `unique_ptr<Queue[]>`? 因爲 `mutex` 不可搬移.
    ```cpp
    // thread_pool.h
    std::unique_ptr<Queue []> work_queues;
    
    // thread_pool.cc 構造函數
    work_queues = std::make_unique<Queue []>(thread_count);
    ```
* **分配策略 (Round-Robin)**:
    ```cpp
    void submit(Func func) {
        work_queues[pos].push(func);
        pos = (pos + 1) % thread_count;
    }
    ```

### 2.3 785：負載不均的“反面教材”

* 展示了靜態分配在面對“長任務”時, 無法利用空閒線程的問題.

---

## 3. 高性能之巔：任務竊取 (790: Work Stealing)

### 3.1 `try_pop` 與 `try_push` (非阻塞接口)

這是實現竊取的關鍵, 不能掛起線程.

```cpp
bool try_pop(T& value) {
    std::lock_guard<std::mutex> lck(mut);
    if (que.empty()) return false;
    value = que.front(); que.pop();
    return true;
}
```

### 3.2 竊取算法 (Work Stealing Strategy)

Worker 不斷檢查自己和別人的隊列, 並防止 CPU 空轉.

```cpp
void thread_pool::worker(int idx) {
    while (true) {
        int visited = 0;
        Func task;
        int i = idx; // 先看自己的
        
        while (!work_queues[i].try_pop(task)) { // 如果拿不到, 就去“偷”
            i = get_random(); 
            if (++visited == thread_count) { // 全找過了都沒有
                std::this_thread::sleep_for(10ms); // 防止 CPU 100% (Hot Loop Avoidance)
                visited = 0; i = idx;
            }
        }
        task();
    }
}
```

### 3.3 進化的 `submit` (隨機分發)

```cpp
void thread_pool::submit(Func func) {
    int i;
    do {
        i = get_random(); // 隨機挑選
    } while (!work_queues[i].try_push(func)); // 沒滿就塞, 滿了換一家
}
```

---

## 4. 關鍵知識點速查表

| 特性 | 實現方式 | 解決的問題 |
| :--- | :--- | :--- |
| **CV + Predicate** | `wait(lock, lambda)` | 虛假喚醒 & CPU 佔用 |
| **多隊列模式** | `unique_ptr<Queue[]>` | 鎖競爭 (Lock Contention) |
| **任務竊取** | `try_pop` + `get_random` | 負載平衡 (Load Balance) |
| **Hot Loop 避免** | `visited` 計數 + `sleep` | 沒任務時的 CPU 浪費 |
| **解構函數** | `join()` 前需有 `done` 標誌 | 優雅關閉 (Graceful Shutdown) |

---
*詳細代碼參考路徑: `750_concurrent_queue` 至 `790_thread_pool_work_stealing_contd`*
