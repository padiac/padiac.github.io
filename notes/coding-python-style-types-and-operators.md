This document compiles the style guidelines, variable typing rules, and sequence operators in Python.

---

## 1. Python Coding Style Guide: PEP 8

### 1.1 What is PEP 8?
* **PEP** stands for **Python Enhancement Proposal**.
* **PEP 8** is the official style guide for writing Python code, focusing on readability and consistency.

### 1.2 PEP 8 Naming Conventions Reference
Python strongly discourages the use of `camelCase` (camel-case) for variables and functions. Instead, it promotes the use of `snake_case` (snake-case).

| Element | Style | Example | Notes |
| :--- | :--- | :--- | :--- |
| **Variables** | `snake_case` (all lowercase, underscores) | `split_string`, `another_string` | Used in escapechar.py |
| **Functions & Methods** | `snake_case` | `calculate_value()`, `get_data()` | Consistently lowercase with underscores |
| **Classes** | `PascalCase` / `CapWords` | `PetShop`, `BinarySearch` | First letter of each word capitalized |
| **Constants** | `UPPER_CASE` | `MAX_LIMIT`, `PI` | Defined at module level, all uppercase |
| **Modules / Packages** | short lowercase | `os`, `sys`, `my_module` | Avoid underscores in packages, modules can use them |

---

## 2. Variables & Dynamic/Strong Typing

### 2.1 Dynamic Typing
Python is a dynamically typed language. A variable can be bound to different types of values during execution:
```python
age = 24         # age is initially an integer (int)
age = "2 years"  # age is now a string (str)
```
The built-in `type()` function returns the current data type of a variable:
* `type(24)` $\rightarrow$ `<class 'int'>`
* `type("Hello")` $\rightarrow$ `<class 'str'>`

### 2.2 Strong Typing & TypeError
Python is a strongly typed language, meaning it does not perform implicit type conversions during operations like concatenation.
* **Error**: `"Tim is " + age + " years old"` throws `TypeError: can only concatenate str (not "int") to str` when `age` is an integer `24`.
* **Fixes**:
  1. Convert manually: `"Tim is " + str(age) + " years old"`
  2. Use f-string: `f"Tim is {age} years old"`
  3. Use comma in print: `print("Tim is", age, "years old")`

---

## 3. Sequence Operators & Precedence

* **`+` (Concatenation)**: Joins two sequences (e.g., `string1 + string2`).
* **Implicit Concatenation**: Consecutive string literals (not variables) are automatically joined:
  ```python
  print("he's " "probably " "fjords")  # Prints: he's probably fjords
  ```
* **`*` (Repetition)**: Repeats a sequence: `"Hello " * 5` prints `Hello` 5 times.
* **Operator Precedence**: `*` has higher precedence than `+`.
  * `"Hello " * (5 + 4)` prints `Hello ` 9 times.
  * `"Hello " * 5 + "4"` prints `Hello ` 5 times and appends `"4"`.
* **`in` / `not in` (Membership)**: Tests if a subsequence exists within a sequence:
  * `"day" in "friday"` $\rightarrow$ `True`
  * `"thur" in "friday"` $\rightarrow$ `False`

### 3.1 Common Sequence Functions & Methods

Python provides built-in functions and methods to query sequences (like strings, lists, and tuples):

* **`min(sequence)`**: Returns the smallest item in the sequence (e.g., `min([2, 4, 6])` $\rightarrow$ `2`).
* **`max(sequence)`**: Returns the largest item in the sequence (e.g., `max([2, 4, 6])` $\rightarrow$ `8`).
* **`len(sequence)`**: Returns the length (number of items) of the sequence (e.g., `len([2, 4, 6])` $\rightarrow$ `3`).
* **`sequence.count(item)`**: Returns the number of times `item` (or substring) appears in the sequence.

#### Crucial Detail: Non-Overlapping Count
For strings, the `.count()` method counts **non-overlapping** occurrences of a substring.

In number_lists.py:
```python
print("mississippi".count("issi"))  # Prints 1
```

* **Walkthrough**: 
  1. Python finds the first `"issi"` at indices `1` to `4` (`m [i s s i] s s i p p i`).
  2. Because it only counts **non-overlapping** matches, it jumps past indices 1-4 and continues searching from index `5` (`ssippi`).
  3. The remaining substring `"ssippi"` does not contain `"issi"`.
  4. The second potential `"issi"` at indices `4` to `7` (`m i s s [i s s i] p p i`) is **skipped** because it overlaps at index 4 (`i`).

---

## 4. String Indexing (Positive & Negative)

In Python, characters within strings can be accessed using 0-based positive indexing or negative indexing.

### 4.1 Indexing Reference Table (for `"Norwegian Blue"`)

| Character | N | o | r | w | e | g | i | a | n | (space) | B | l | u | e |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Positive Index** | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
| **Negative Index** | -14 | -13 | -12 | -11 | -10 | -9 | -8 | -7 | -6 | -5 | -4 | -3 | -2 | -1 |

### 4.2 Index Conversion Formula
$$\text{Negative Index} = \text{Positive Index} - \text{String Length}$$
For example, to find the letter `'w'` in `"Norwegian Blue"` (length 14):
* Positive index: `parrot[3]`
* Negative index: `parrot[3 - 14]` $\rightarrow$ `parrot[-11]`

---

## 5. String Slicing (Basic, Backwards & Safety)

### 5.1 Basic Slicing Syntax
$$\text{string}[\text{start} : \text{stop} : \text{step}]$$
* **`start`**: Index where slicing starts (**inclusive**). Defaults to `0`.
* **`stop`**: Index where slicing stops (**exclusive**). Defaults to length of string.
* **`step`**: Number of characters to skip. Defaults to `1`.

* **Examples**:
  * `parrot[0:6]` $\rightarrow$ `"Norweg"` (indices 0 to 5)
  * `parrot[10:]` $\rightarrow$ `"Blue"` (from index 10 to end)
  * `parrot[:]` $\rightarrow$ copy of the entire string

### 5.2 Slicing Backwards (Negative Step)
When **`step` is negative**, slicing moves from **right to left**:
* **String Reversal**: `letters[::-1]` reverses the entire string (e.g., `"zyx...a"`).
* `letters[4::-1]` $\rightarrow$ `"edcba"` (from index 4 down to 0).
* `letters[:-9:-1]` $\rightarrow$ `"zyxwvuts"` (the last 8 characters in reverse order).

### 5.3 Slice vs. Index Safety on Empty Strings
> [!TIP]
> Accessing index `0` of an empty string (`letters[0]`) raises an `IndexError`. Using a slice instead (`letters[:1]`) returns an empty string `""` safely without raising an error.

---

## 6. String Formatting Methods

Python provides three ways to format strings:

### 6.1 Positional Replacement Fields (`.format()`)
```python
age = 24
print("My age is {0} years".format(age))
```
* Arguments are mapped by index. An index can be reused multiple times:
  ```python
  # Reuses parameter 2 (31) and parameter 1 (30) multiple times
  print("Jan: {2}, Feb: {0}, Mar: {2}, Apr: {1}".format(28, 30, 31))
  ```

### 6.2 F-Strings (Python 3.6+) (Preferred)
Prepended with `f` or `F`, variables/expressions are written directly inside braces `{}`:
```python
print(f"My age is {age} years")
print(f"Next year you will be {age + 1}")
```
* **Pros**: Best readability, support for arbitrary expressions (e.g. methods `{name.upper()}`), and significantly faster execution (up to 2-3x faster than `.format()`).

### 6.3 Legacy `%` Interpolation (printf-style)
```python
print("My age is %d %s" % (age, major))
```
* **Placeholders**: `%d` for integer, `%s` for string, `%f` for float. Arguments must be enclosed in a tuple. This is legacy syntax and should be avoided in new code.

---

## 7. Formatting Alignment & Precision

For both `.format()` and `f-strings`, the layout rules after the colon `:` are identical:

### 7.1 Field Alignment
* **`<`**: Left alignment (e.g. `{val:<3}`).
* **`^`**: Centered alignment (e.g. `{val:^4}`).
* **Default / `>`**: Right alignment (default for numbers).

### 7.2 Float Precision (Example: Pi Approximation)
* **`{0:12f}`**: Floating-point number with a total field width of 12 (defaults to 6 decimal places).
* **`{0:12.50f}`**: 50 decimal places. The precision overrides the width 12.
* **`{0:62.50f}`**: 50 decimal places, total width 62.
  * **Padding Calculation**: A float with 50 decimal places takes exactly 52 characters (1 integer + 1 decimal point + 50 decimals). With width 62, Python pads **`62 - 52 = 10`** spaces on the left (due to default right alignment).

---

## 8. Special String Literals

### 8.1 Triple Quotes (`"""` or `'''`)
* **Quotes Escaping**: Wrapping a string in triple quotes allows you to write single and double quotes directly:
   ```python
   print("""The owner said "No, no, 'e's resting". """)
   ```
* **Multi-line Strings**: Spans multiple lines naturally without `\n`.
* **Docstrings**: Standard conventions for documenting modules, classes, and functions.

### 8.2 Raw Strings (`r"..."`)
Prefixing a string with `r` or `R` suppresses escape sequences:
* `print(r"C:\Users\timbuchalka\notes.txt")` prints literal backslashes without escaping. Ideal for file paths and regex.
