# C++ 语法备忘录（LeetCode 刷题随记）

> 用途：刷 LeetCode 时，记录容易忘或和直觉不一致的 C++ 语法点  
> 原则：只记用法 + 关键差异，不写长解释

---

## 1. ListNode 指针与 new
```cpp
ListNode* node = new ListNode(5);
```
- `new` 返回对象地址（指针）
- LeetCode 中通常不需要手动 `delete`

---

## 2. unordered_set 插入 / 删除
```cpp
unordered_set<int> s;
s.insert(10);
s.erase(10);
```

---

## 3. 二维 vector 初始化（DP 常用）
```cpp
vector<vector<bool>> dp(n, vector<bool>(n, false));
```
- 第一维大小：`n`
- 第二维大小：`n`
- 初始值：`false`

---

## 4. 容器大小
```cpp
v.size();
```
- 返回类型是 `size_t`

---

## 5. string::substr
```cpp
s.substr(start_index, length);
```
- 第一个参数：起始 index
- 第二个参数：长度（不是结束位置）

---

## 6. INT_MIN / INT_MAX
```cpp
#include <climits>

INT_MIN;
INT_MAX;
```

---

## 7. string::starts_with（C++20）
```cpp
s.starts_with("leet");
```

---

## 8. sort
```cpp
sort(v.begin(), v.end());
sort(v.begin(), v.end(), greater<int>());
```

---

## 9. push_back + 初始化列表
```cpp
ret.push_back({nums[i], nums[j], nums[k]});
```

---

## 10. stack 常用操作
```cpp
stack<int> st;
st.push(x);
st.pop();
st.top();
st.empty();
```

---

## 11. 函数参数尽量用引用
```cpp
void f(vector<int>& v);
void g(const vector<int>& v);
```

---

## 12. vector 没有 add
```cpp
v.push_back(x);
v.emplace_back(x);
```

---

## 13. vector.insert
```cpp
v.insert(v.begin() + i, x);
```

---

## 14. map 遍历取 second
```cpp
for (auto& item : mp) {
    ret.push_back(item.second);
}
```

---

## 15. map 遍历（只记这一种）
```cpp
for (auto& kv : mp) {
    kv.first;
    kv.second;
}
```

---

## 16. stringstream 按字符切分字符串
```cpp
#include <sstream>

stringstream ss(path);
string tmp;

while (getline(ss, tmp, '/')) {
    // tmp 是每一段
}
```

---

## 17. 字符判断与大小写转换（<cctype>）
```cpp
#include <cctype>

isalpha(c);
isdigit(c);
isalnum(c);
tolower(c);
toupper(c);
```

---

> 后续新语法点：按编号继续追加
