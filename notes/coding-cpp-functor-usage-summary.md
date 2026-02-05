Functor (仿函数), 即**函数对象 (Function Object)**, 是指重载了 `operator()` 操作符的类或结构体对象. 它们可以像函数一样被调用, 但比普通函数功能更强大、更灵活.

## 1. 核心概念

Functor 本质上是一个对象, 但它表现得像一个函数.

```cpp
class Add {
public:
    // 重载 () 操作符
    int operator()(int a, int b) {
        return a + b;
    }
};

Add add;
int result = add(3, 4); // 像调用函数一样使用对象
```

## 2. 为什么使用 Functor？(优势)

* **保持状态 (Stateful)**: 普通函数通常没有状态（除非使用静态变量），而 Functor 可以通过成员变量保存状态。
* **性能优势**: 编译器可以很容易地内联 (Inline) Functor 的调用，而函数指针通常难以内联。
* **类型识别**: 每个 Functor 都有自己的类型，这在模板编程中非常有用。

## 3. 常见用法场景

### 场景 A: 保持状态 (Stateful Functor)

这是 Functor 最强大的特性之一。可以通过构造函数初始化状态，并在后续调用中利用甚至修改这个状态。

```cpp
class Multiplier {
    int factor; // 内部状态
public:
    Multiplier(int f) : factor(f) {}
    
    int operator()(int x) {
        return x * factor; // 使用内部状态
    }
    
    void setFactor(int f) { factor = f; } // 修改状态
};

Multiplier mult(5);
std::cout << mult(10); // 输出 50
```

### 场景 B: 配合 STL 算法 (作为谓词)

STL 算法（如 `count_if`, `find_if`, `remove_if`）经常需要一个“谓词” (Predicate) 来判断条件。

```cpp
class GreaterThan {
    int threshold;
public:
    GreaterThan(int t) : threshold(t) {}
    
    bool operator()(int value) {
        return value > threshold;
    }
};

std::vector<int> nums = {1, 5, 10, 20};
// 统计大于 10 的元素
int count = std::count_if(nums.begin(), nums.end(), GreaterThan(10));
```

### 场景 C: 自定义排序规则

`std::sort` 或 `std::set`/`std::map` 需要比较逻辑时，Functor 是标准做法。

```cpp
struct CompareDescending {
    bool operator()(int a, int b) {
        return a > b; // 降序
    }
};

std::sort(nums.begin(), nums.end(), CompareDescending());
```

### 场景 D: 作为线程任务 (Thread Task)

`std::thread` 可以接受一个 Functor 作为线程入口。这是一种很好的将“数据”和“逻辑”打包传给线程的方式。

```cpp
class ThreadTask {
public:
    void operator()(int x) {
        std::cout << "Thread processing: " << x << std::endl;
    }
};

ThreadTask task;
std::thread t(task, 100); // 在新线程中执行 task(100)
t.join();
```

## 4. 总结 (Functor vs Lambda)

现代 C++ (C++11及以后) 引入了 **Lambda 表达式**, 它本质上是**匿名 Functor** 的语法糖.

* **简单的、一次性的逻辑**: 推荐用 Lambda.
* **复杂的、需要复用的、有大量内部状态的逻辑**: 显式定义的 Functor 类依然更有优势, 结构更清晰.

---
*参考代码: `010_launching_a_thread/functor_usage.cc`*
