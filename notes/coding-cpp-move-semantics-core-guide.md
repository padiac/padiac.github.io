在 C++ 中, `std::move` 配合**不同的函数参数写法**, 会有完全不同的行为. 这是移动语义中最关键的概念.

## 核心区别：两种“接收”参数的方式

### 1. 彻底接管模式 (Pass by Value / Sink) —— *推荐*

这是现代 C++ 处理资源所有权转移（如 `std::thread`, `std::unique_ptr`, `std::vector`）的标准写法.

```cpp
// 场景：「我要这块资源，原来的不再需要了」
// 签名特点：参数通过值传递 (无 & 或 &&)
void func(std::string str) {
    // str 是一个全新的独立对象
    // 它通过“移动构造函数”从外面的变量那里“偷”来了资源
}

int main() {
    std::string s = "Hello World";
    // 行为描述：
    // 1. std::move(s) 将 s 标记为右值。
    // 2. 为了构造参数 str，编译器调用 string 的移动构造函数。
    // 3. 在 func 执行的第一行之前，s 的内容已经被偷走（变空）。
    func(std::move(s)); 
}
```

* **发生时间**: 进入函数的一瞬间 (构造参数时).
* **动作**: 调用 `T(T&&)` 移动构造函数.
* **原变量状态**: **立刻变空** (对于 `string/vector/thread`), 确保资源只存在于函数内.

### 2. 引用模式 (Pass by Rvalue Reference) —— *底层或特定场景*

这种写法并不转移所有权, 仅仅是传递一个“具名右值引用”.

```cpp
// 场景：「我要引用这个临时对象，或者对其进行手动修改」
// 签名特点：参数通过右值引用传递 (带 &&)
void func(std::string&& str) {
    // str 只是外面那个变量的这一层皮（引用/别名）。
    // 此时并没有发生任何移动或拷贝。
    
    // 如果函数体内什么都不做，外面那个变量毫发无损！
    // 只有显式操作才会影响原变量，例如：
    // str.clear();  // 手动清空原变量
    // std::string s2 = std::move(str); // 手动偷走资源
}

int main() {
    std::string s = "Hello World";
    // 行为描述：
    // 1. std::move(s) 将 s 标记为右值。
    // 2. 参数 str 绑定到 s 上。
    // 3. 进入函数时，s 的内容还在！并没有自动变空。
    func(std::move(s));
}
```

* **发生时间**: 不自动发生, 取决于函数体内代码.
* **动作**: 仅仅传了个引用.
* **原变量状态**: **保持原样** (除非函数内部显式清空或窃取).

---

## 经典案例分析

### 案例 A：std::thread (独占型资源)

`std::thread` 是一种不可复制的资源.
* **推荐写法**: `void func(std::thread t)` (按值传递)
    * `main` 把线程控制权彻底移交. `main` 的句柄立刻失效, `func` 里的 `t` 成为唯一拥有者, 负责 `join()`. 这是最安全、最符合语义的写法.

### 案例 B：std::string / std::vector (深拷贝型资源)

* **推荐写法**: `void func(std::string t)` (按值传递)
    * 通过移动构造函数偷走堆内存指针.
    * 如果用引用模式 `string&&`, 容易忘记手动移走资源, 导致资源生命周期混乱.

### 案例 C：int (无资源类型)

* **本质**: `int` 没有指针可偷.
* **行为**: 即使使用了 `std::move`, 或者写成了移动构造的形式, 底层也只能执行**拷贝 (Copy)**.
* **误区**: 以为 `move(int)` 会把原值清零. 事实是原值完全不变, 因为没有“偷”这个动作的空间.

---

## 关键规则：具名右值引用是左值

这是一个极其反直觉但至关重要的规则:

> **如果你给一个右值引用起了名字, 那它在当前作用域内就是一个左值.**

```cpp
// Example of a function argument which is an rvalue reference
#include <iostream>

// The caller's object will be moved into x
void func(int&& x)
{
	std::cout << "Rvalue reference" << std::endl;
}

void func2(int& x)
{
	std::cout << "Lvalue reference" << std::endl;
}

int main()
{
	func(2);

	int y = 2;
	//func(y);      // Error! Must be a moveable rvalue
	func(std::move(y));

	func2(y);
}
```

**为什么这样设计？**
为了安全. 如果 `x` 自动被视为右值, 可能在第一行代码就被“移走”空了, 导致后面再用 `x` 时出现未定义行为. 编译器强制要求: **如果你想再次转移所有权, 必须显式地再次写 `std::move(x)`**.
