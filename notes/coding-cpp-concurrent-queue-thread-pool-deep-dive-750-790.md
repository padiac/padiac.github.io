## 1. 并发队列的基础演进 (750-760)

### 1.1 从“报错”到“等待” (750 vs 760)

* **750 基础型**:
    ```cpp
    // 逻辑: 不等待, 直接抛异常
    void push(T value) {
        std::lock_guard<std::mutex> lck(mut);
        if (que.size() >= max) throw concurrent_queue_full();
        que.push(value);
    }
    ```
* **760 混合阻塞型 (`concurrent_queue_cv`)**:
    ```cpp
    // Pop 完全阻塞, Push 轮询 (混合模式)
    void push(T value) {
        std::unique_lock<std::mutex> uniq_lck(mut);
        while (que.size() > max) { // 轮询
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

### 1.2 条件变量 (CV) 的精确使用

* **虚假唤醒 (Spurious Wakeup)**: 使用 Predicate Lambda 防止线程意外醒来.
* **通知策略**: 使用 `notify_one()` 减少线程竞争 (惊群效应).

---

## 2. 线程池：单队列 vs 多队列 (770-785)

### 2.1 770：全局共享队列 (Lock Contention)

* 所有 Worker 共用同一个队列, 争夺一把锁. 适用于任务较重的场景.

### 2.2 780：分而治之 (Multiple Queues per Thread)

* **内存管理**: 为什么用 `unique_ptr<Queue[]>`? 因为 `mutex` 不可搬移.
    ```cpp
    // thread_pool.h
    std::unique_ptr<Queue []> work_queues;
    
    // thread_pool.cc 构造函数
    work_queues = std::make_unique<Queue []>(thread_count);
    ```
* **分配策略 (Round-Robin)**:
    ```cpp
    void submit(Func func) {
        work_queues[pos].push(func);
        pos = (pos + 1) % thread_count;
    }
    ```

### 2.3 785：负载不均的“反面教材”

* 展示了静态分配在面对“长任务”时, 无法利用空闲线程的问题.

---

## 3. 高性能之巅：任务窃取 (790: Work Stealing)

### 3.1 `try_pop` 与 `try_push` (非阻塞接口)

这是实现窃取的关键, 不能挂起线程.

```cpp
bool try_pop(T& value) {
    std::lock_guard<std::mutex> lck(mut);
    if (que.empty()) return false;
    value = que.front(); que.pop();
    return true;
}
```

### 3.2 窃取算法 (Work Stealing Strategy)

Worker 不断检查自己和别人的队列, 并防止 CPU 空转.

```cpp
void thread_pool::worker(int idx) {
    while (true) {
        int visited = 0;
        Func task;
        int i = idx; // 先看自己的
        
        while (!work_queues[i].try_pop(task)) { // 如果拿不到, 就去“偷”
            i = get_random(); 
            if (++visited == thread_count) { // 全找过了都没有
                std::this_thread::sleep_for(10ms); // 防止 CPU 100% (Hot Loop Avoidance)
                visited = 0; i = idx;
            }
        }
        task();
    }
}
```

### 3.3 进化的 `submit` (随机分发)

```cpp
void thread_pool::submit(Func func) {
    int i;
    do {
        i = get_random(); // 随机挑选
    } while (!work_queues[i].try_push(func)); // 没满就塞, 满了换一家
}
```

---

## 4. 关键知识点速查表

| 特性 | 实现方式 | 解决的问题 |
| :--- | :--- | :--- |
| **CV + Predicate** | `wait(lock, lambda)` | 虚假唤醒 & CPU 占用 |
| **多队列模式** | `unique_ptr<Queue[]>` | 锁竞争 (Lock Contention) |
| **任务窃取** | `try_pop` + `get_random` | 负载均衡 (Load Balance) |
| **Hot Loop 避免** | `visited` 计数 + `sleep` | 没任务时的 CPU 浪费 |
| **析构函数** | `join()` 前需有 `done` 标志 | 优雅关闭 (Graceful Shutdown) |

---
*详细代码参考路径: `750_concurrent_queue` 至 `790_thread_pool_work_stealing_contd`*
