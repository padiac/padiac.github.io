This document details Python function parameters, docstrings, type annotations, and APIs.

---

## 1. String Utilities: `isalnum()` & `center()`

### 1.1 `str.isalnum()`
* **Definition**: Checks if all characters in the string are alphanumeric (either letters or numbers).
* **Behavior**: Returns `True` if the string contains only letters (`a-z`, `A-Z`) and numbers (`0-9`), and contains at least one character. Returns `False` if there are any spaces, punctuation marks, or special symbols.
* **Use Case (Palindrome Check)**: In functions.py, `isalnum()` is used in a loop to filter out spaces and punctuation from a sentence, leaving only alphanumeric characters to safely check if the sentence is a palindrome.

### 1.2 `str.center(width)`
* **Definition**: Centers the string inside a field of the specified `width`, padding it on both sides with spaces (by default).
* **Use Case (Banner Text Alignment)**: In banner.py, `.center(screen_width - 4)` is used to center the content text within the border:
  ```python
  centred_text = text.center(screen_width - 4)
  output_string = "**{0}**".format(centred_text)
  ```
  This guarantees the total output width is exactly `screen_width` regardless of the text length.

---

## 2. Functions: Default Values, Keyword Arguments & Positional-Only Parameters

### 2.1 Default Parameter Values
In Python, parameters can have default values defined using `parameter=default_value` (e.g. `def banner_text(text=" ", screen_width=80)`). If an argument is omitted during the call, its default value is used.

### 2.2 Keyword Arguments
You can specify arguments by parameter name during the call (e.g. `banner_text(screen_width=60)`). This allows you to skip positional arguments that have default values, and pass values only to specific parameters.

### 2.3 Positional-Only Parameters (`/`)
* **PEP 570 (Python 3.8+)** introduces the `/` separator in a function's parameter list:
  ```python
  def func(posonly1, posonly2, /, positional_or_keyword):
      pass
  ```
* **Rules**: Any parameters to the **left** of `/` must be passed **positionally** (e.g. `func(1, 2, 3)` or `func(1, 2, positional_or_keyword=3)`). Passing them by name (e.g. `func(posonly1=1, ...)`) raises a `TypeError`.
* **Purpose**: Prevents callers from relying on parameter names (allowing developers to rename parameters without breaking backwards compatibility) and aligns Python functions with native C-implemented built-ins.

---

## 3. Documenting Code: Docstrings

### 3.1 What is a Docstring?
A **Docstring** (Documentation String) is a triple-quoted string `"""` placed as the first statement in a module, class, or function definition.
* **Purpose**: Explains what the code does, its parameters, and its return values.
* **Usage**: Can be displayed using `help(function_name)` or viewed as hover help/tooltips inside modern IDEs.

### 3.2 Structure of a Docstring
A standard Docstring typically consists of:
1. A **one-line summary** starting with a capital letter and ending with a period.
2. A **blank line** followed by a **detailed description** of the function's behavior.
3. Parameter and return value annotations using standard formats (e.g. Sphinx/reST format `:param:` and `:return:`, Google format `Args:` and `Returns:`, or NumPy format).

### 3.3 Example
In guessinggame.py:
```python
def get_integer(prompt):
    """
    Get an integer from Standard Input (stdin).

    The function will continue looping, and prompting
    the user, until a valid `int` is entered.

    :param prompt: The String that the user will see, when
        they're prompted to enter the value.
    :return: The integer that the user enters.
    """
```

---

## 4. Parameter Types & Arguments Unpacking

In Python, a function definition can accept a mixture of different parameter types. The order and syntax of these parameters define how arguments are captured.

### 4.1 Parameter Types Reference Code

In parameter_types.py:
```python
def func(p1, p2, *args, k, **kwargs):
    print("positional-or-keyword:...{}, {}".format(p1, p2))
    print("var-positional (*args):..{}".format(args))
    print("keyword:.................{}".format(k))
    print("var-keyword:.............{}".format(kwargs))

func(1, 2, 3, 4, 5, 9, k=6, key1=7, key2=8)
```

### 4.2 Parameter Classifications

1. **Positional-or-Keyword (`p1, p2`)**:
   Standard parameters that can be bound either by position or explicitly by name. In the call above, `p1` gets `1` and `p2` gets `2`.
2. **Var-Positional (`*args`)**:
   Prefixing a parameter with a single asterisk `*` tells Python to collect all remaining, unnamed positional arguments into a single **tuple**. In the call, `3, 4, 5, 9` are collected into `args` as `(3, 4, 5, 9)`.
3. **Keyword-Only (`k`)**:
   Any parameter placed after `*args` (or after a bare `*`) is a keyword-only parameter. It cannot be passed positionally. When calling the function, you must pass it as `k=value`.
4. **Var-Keyword (`**kwargs`)**:
   Prefixing a parameter with a double asterisk `**` collects all remaining, unnamed keyword arguments into a single **dictionary**. In the call, `key1=7` and `key2=8` are collected into `kwargs` as `{'key1': 7, 'key2': 8}`.

### 4.3 Star `*` Operator: Packing vs. Unpacking
The star `*` operator has two complementary behaviors depending on where it is used:
* **Packing** (In Function Definitions): Gathers individual positional arguments into a tuple (e.g. `*args`).
* **Unpacking** (In Function Calls): Disassembles a list/tuple/set and passes its elements as individual positional arguments to the function.
  * **Example**:
    ```python
    numbers = (0, 1, 2, 3, 4, 5)
    print(*numbers, sep=";")  # Unpacks numbers, prints: 0;1;2;3;4;5
    ```

---

## 5. Generators & The `yield` Keyword

A **Generator** is a special type of function that returns an iterator, producing a sequence of values on the fly instead of returning a finished data structure all at once.

### 5.1 The `yield` Keyword
* **`return` vs. `yield`**:
  * `return` returns a value and completely terminates the function's execution, freeing its local stack/namespace.
  * `yield` returns a value to the caller, but **pauses (freezes) the function's execution**, retaining all variables and stack state. When the generator is called again, it resumes execution from the line immediately following the `yield`.
* **Memory Efficiency**: Since generators compute values one-by-one on demand (lazy evaluation/惰性求值), they require $O(1)$ memory. They are ideal for processing extremely large data streams or infinite series where storing a list in memory is impossible.

### 5.2 Generator Type Annotations (Python 3.5+)
A generator function uses the `Generator` type hint from the `typing` module:
```python
from typing import Generator

def my_generator(n: int) -> Generator[YieldType, SendType, ReturnType]:
    pass
```
* **`YieldType`**: The type of value yielded by the function (e.g. `int` in `squares_generator`).
* **`SendType`**: The type of value that can be sent *into* the generator via `.send()`. If not used, this is `None`.
* **`ReturnType`**: The value returned by the generator when it terminates (via a standard `return` statement). If it doesn't return anything explicitly, this is `None`.

### 5.3 Generator Objects vs. Collecting Items
When you call a generator function, it does **not** execute the code or return a list; it returns a **Generator Object**:
```python
gen = squares_generator(1000)
print(gen)  # Output: <generator object squares_generator at 0x...>
```
To run the code and collect the generated values, you must iterate over it:
1. **Explicit conversion** to collections (e.g. `list(gen)` or `set(gen)`).
2. **For loop** iteration (`for x in gen:`).
3. **Manual iteration** using `next(gen)`.

### 5.4 Generator Exhaustion vs. Reusable Iterables
* **Iterator Exhaustion**: A generator object is a single-use **Iterator**. It maintains an internal pointer to its execution state. Once it runs to completion (i.e. the function exits or returns, raising `StopIteration`), the generator is **exhausted**. Attempting to iterate over the same generator object a second time will yield nothing because it cannot rewind.
  * **Fix**: To reuse the generator logic, a new generator object must be created by calling the generator function again (e.g. `gen = squares_generator(1000)`).
* **Iterables (e.g. `range()`)**: Built-in lazy iterables like `range(5)` are **not** single-pass iterators. They are **Iterable** sequence containers. Every time they are iterated (e.g., in a `for` loop), Python calls `iter(range(5))` under the hood, creating a fresh, new iterator starting from 0. Therefore, a `range` object can be iterated multiple times.

### 5.5 Infinite Generators & the `next()` Function
Generators are capable of representing infinite mathematical series or streams using infinite loops (e.g. `while True:`) because they evaluate lazily.
* **The `next()` Function**: The built-in `next(generator)` function manually retrieves the next yielded value by advancing the generator to its next `yield` statement.
* **Example (Fibonacci Sequence)**:
  ```python
  def fibonacci():
      current, previous = 0, 1
      while True:
          yield current
          current, previous = current + previous, current
  ```
  Since `yield` pauses execution, calling `next(fib)` yields the next number in the infinite sequence on demand, without getting stuck in a dead loop or running out of memory.

### 5.6 Compound Generators & Alternating Series (Leibniz Pi Formula)
Generators can be combined (nested) to compute complex mathematical sequences.
* **Leibniz Formula for Pi**:
  `pi = 4 * (1 - 1/3 + 1/5 - 1/7 + 1/9 - 1/11 ...)`
* **Double Yield Pattern**: To alternate between addition and subtraction terms, a generator can contain multiple `yield` statements within a single loop:
  ```python
  def pi_series():
      odds = oddnumbers()  # A generator yielding 1, 3, 5, 7, ...
      approximation = 0
      while True:
          # Phase 1: Addition (+)
          approximation += (4 / next(odds))
          yield approximation
          # Phase 2: Subtraction (-)
          approximation -= (4 / next(odds))
          yield approximation
  ```
  - The first call to `next()` executes up to the first `yield` (adding a term).
  - The second call to `next()` resumes and executes up to the second `yield` (subtracting a term).
  - The third call loops back to the start of the `while` block, repeating the pattern.

---

## 6. Scope, Namespaces, and Nested Functions

In Python, every name (variable, function, module) belongs to a namespace, which governs how Python resolves names.

### 6.1 What is a Namespace?
A **Namespace** is a mapping from names (strings) to objects. Under the hood, Python implements namespaces as standard dictionaries.

#### The LEGB Scope Resolution Rule
When a variable name is referenced, Python searches namespaces in this order:
1. **L (Local)**: Names defined inside the current function.
2. **E (Enclosing)**: Names defined in any enclosing (outer) functions (relevant to nested functions/closures).
3. **G (Global)**: Names defined at the top-level of the current module/file. Accessible via `globals()`.
4. **B (Built-in)**: Python's built-in names (e.g., `print`, `len`, `int`, exceptions). Accessible via `dir(__builtins__)`.

---

### 6.2 Global Namespace (`globals()`) vs. Local Namespace (`locals()`)
* **`globals()`**: Returns the actual dictionary representing the current global namespace.
  * Modifying keys in the dictionary returned by `globals()` directly changes or creates global variables.
  * **Introspection via `dir()`**: At the top level of a module, `dir()` returns the sorted list of keys in the global namespace (i.e. `sorted(list(globals().keys()))`).
* **`locals()`**: Returns a dictionary representing the current local namespace (inside a function).
  * Variables are added to `locals()` dynamically as they are bound.

#### ⚠️ Dictionary Changed Size During Iteration
A strict rule in Python is that **you cannot modify a dictionary's keys (add or remove items) while iterating over it**. 
Because `globals()` is a live dictionary, trying to iterate over it directly in a loop that creates new variables (like `for name, obj in globals()`) will crash the program with `RuntimeError: dictionary changed size during iteration`. 
* **Fix**: Iterate over a static snapshot/copy of the dictionary: `globals().copy()`.

---

### 6.3 Nested Functions, Closures, and Shadowing
Python allows defining functions inside other functions.

#### 1. Nested Functions & Closures
An inner function can read variables defined in its outer enclosing function. This state preservation is called a **closure**.
* **Example**:
  ```python
  def greet_pythons(items: list) -> None:
      greeting = 'Hello'  # Enclosing variable

      def make_greeting(item: str) -> str:
          return f'{greeting} {item}'  # Accesses enclosing variable

      for item in items:
          print(make_greeting(item))
  ```

#### 2. Variable Shadowing (遮蔽)
If an inner function assigns a value to a variable (e.g., `greeting = 'Hi'`), Python automatically treats it as a **new local variable** inside the inner function.
* It does **not** modify the outer function's variable.
* The local variable "shadows" (masks) the outer variable.
* Use `id(variable)` to confirm they are distinct objects in memory.

---

### 6.4 Importing Modules: Side Effects & Guards
* **Execution on Import**: When a module is imported for the first time, Python runs its code from top to bottom. Any top-level code (like `print()`) will execute immediately, causing unexpected side effects.
* **`if __name__ == '__main__':` Guard**: 
  To prevent top-level code from running during an import, wrap it in this condition. 
  * If the file is run directly, `__name__` is set to `'__main__'` and the block executes.
  * If the file is imported, `__name__` is the module's name, so the block is skipped.

---

### 6.5 First-Class Functions
In Python, functions are first-class objects. You can:
* Store them in dictionaries:
  ```python
  supported_hashes = {
      'md5': hashlib.md5,
      'sha256': hashlib.sha256,
  }
  ```
* Retrieve and call them dynamically:
  ```python
  hash_algorithm = supported_hashes.get(algorithm)
  file_hash = hash_algorithm(contents).hexdigest()

---

## 7. Graphics & Geometry: Turtle Concepts and Math

This section covers the coordinates, trigonometry, and pattern-generation concepts used during our Turtle graphics investigations.

### 7.1 Turtle Default State & Coordinate Tracing
* **Starting Position**: When a Turtle window opens, the drawing pen starts at coordinates `(0, 0)` (center of the screen).
* **Starting Direction**: The default heading is **East** (0 degrees).
* **State Restoration**: A key practice in graphic APIs is ensuring that custom drawing functions leave the pen in the exact same position and orientation as they found it.
  * *Example*: Drawing a square using `for side in range(4): forward(length); right(90)` moves the turtle around the perimeter and returns it to `(0, 0)` facing **East** because it completes exactly 360 degrees of rotation ($4 \times 90^\circ$).

---

### 7.2 The Math of a Circumscribed Circle
To draw a circle that perfectly encloses (circumscribes) a square:
1. **Calculate the Radius ($R$)**:
   The distance from the center of a square with side $L$ to any of its corners is the radius of the circumscribed circle. By trigonometry:
   $$R = \frac{L}{\sqrt{2}} = L \times \cos(45^\circ) \approx 0.7071 \times L$$
   In Python, this is calculated as:
   ```python
   radius = length * cos(radians(45))
   ```

2. **Locate the Center using Turtle Angles**:
   * Python's `circle(radius)` function draws a circle whose center is exactly `radius` units **to the left** (90 degrees counter-clockwise) of the turtle's current direction.
   * If the turtle is at the top-left corner `(0, 0)` facing **East**, turning right by 135 degrees makes the turtle face **Southwest** (225 degrees).
   * The direction 90 degrees to the left of **Southwest** is **Southeast** (135 degrees), which points directly at the center of the square.
   * Thus, drawing a circle of the computed $R$ from this direction perfectly centers the circle on the square, touching all four corners.

---

### 7.3 Generating Spirographs & Symmetric Patterns
* **Radial Symmetry**: You can generate complex, mandala-like symmetric patterns by repeatedly drawing a shape, rotating the canvas slightly, and drawing it again.
* **The $360^\circ$ Loop Rule**: To make the pattern complete a perfect loop without overlapping unevenly, the number of repetitions multiplied by the rotation angle must equal exactly $360^\circ$.
  * *Example*: Looping 72 times and rotating 5 degrees each time:
    ```python
    for s in range(72):
        encircled_square(120)
        left(5)  # 72 * 5 = 360 degrees
    ```
* **Performance Tuning (Tracer)**:
  * Turtle animations can be very slow for complex loops.
  * Use `Screen().tracer(0)` to disable rendering animations, draw everything in memory, and then call `Screen().update()` to show the final shape instantly.

---

### 7.4 System Dependencies: Tkinter on Linux/WSL
* **Error**: `ModuleNotFoundError: No module named 'tkinter'`
* **Cause**: Python's `turtle` library relies on `tkinter` (a GUI library). On Debian/Ubuntu Linux distributions (including WSL), `tkinter` is not bundled with the default Python installation to keep the minimal footprint small.
* **Solution**: Install it manually via the package manager:
  ```bash
  sudo apt-get update && sudo apt-get install -y python3-tk
  ```

---

## 8. Datetime, Timedelta, and Timezones

This section covers Python's tools for handling dates, times, durations, and timezone calculations.

### 8.1 Dates and Times Basics
* **`datetime.date`**: Represents a calendar date (Year, Month, Day).
  * Accessing numeric parts: `date.year`, `date.month`, `date.day` return integers.
  * Current date: `datetime.date.today()`.
* **`datetime.time`**: Represents a time of day (Hour, Minute, Second, Microsecond). 
  * Omitted arguments default to `0`.
* **Parsing ISO Strings**:
  Use `fromisoformat(string)` to parse standard ISO-formatted text into Python objects:
  ```python
  _time = time.fromisoformat('11:15:00')
  _date = date.fromisoformat('2022-05-10')
  ```

---

### 8.2 Time Intervals (`timedelta`)
`datetime.timedelta` represents durations or differences between two dates/times.
* **Normalization**: Python automatically converts smaller units (like hours/minutes) into days, seconds, and microseconds internally.
  * *Example*: `timedelta(days=15, hours=48)` normalizes to `17 days, 0:00:00`.
* **Representation**: `repr(timedelta(hours=2))` outputs `datetime.timedelta(seconds=7200)`.

---

### 8.3 Timezones with `zoneinfo`
A `datetime` object is **naive** (no timezone info) by default. To make it **aware**, we use the `zoneinfo` module (standard since Python 3.9).

#### 1. Compatibility Import Guard
For Python versions below 3.9, fall back to `backports.zoneinfo`:
```python
try:
    import zoneinfo
except ImportError:
    from backports import zoneinfo
```

#### 2. Converting Timezones
* **Get Aware UTC Time**: `datetime.now(timezone.utc)` (returns UTC time with `+00:00` offset).
* **Convert to Local**: `utc_time.astimezone()` (automatically reads local system offset).
* **Convert to Specific Zone**: 
  ```python
  ny_tz = zoneinfo.ZoneInfo('America/New_York')
  ny_time = utc_time.astimezone(ny_tz)  # Converted using IANA rules
  ```
* **Implicit Conversion**: If `.astimezone(tz)` is called on a **naive** datetime, Python implicitly assumes the input is in the system's local timezone, binds it, and converts it to the target `tz`.

---

### 8.4 The Daylight Saving Time (DST) Arithmetic Bug
When performing calculations that cross a DST transition boundary (e.g., when the clock skips forward by 1 hour), directly adding a `timedelta` to a local timezone-aware datetime introduces a bug.

#### The Problem: Local Arithmetic
```python
# UK time starts at 00:25 GMT (+00:00) on the DST transition day
uk_time = datetime(2022, 3, 27, 0, 25, tzinfo=uk_tz)
# Directly adding 125 minutes
uk_time = uk_time + td  # Results in 02:30 BST (+01:00)
```
* *Why it fails*: Python adds 125 minutes directly to the local wall-clock time (`00:25` $\rightarrow$ `02:30`). Because of the skipped hour (`01:00` $\rightarrow$ `02:00`), the local clock reading `02:30` is **physically only 65 minutes** after `00:25`. The physical duration is wrong.

#### The Solution: The "UTC Arithmetic" Golden Rule
To ensure physical durations are correct when doing calculations, follow this sequence:
1. **Convert to UTC**: `utc_time = uk_time.astimezone(timezone.utc)`
2. **Add/Subtract timedelta in UTC**: `utc_time = utc_time + td`
3. **Convert back to local timezone**: `uk_time = utc_time.astimezone(uk_tz)`

This ensures the absolute time elapsed is exactly 125 minutes, and the local wall clock correctly adjusts to `03:30 BST` (+01:00) to account for the DST jump.

---

## 9. Object-Oriented Programming (OOP): Fundamentals & Design Patterns

This section details classes, variables, encapsulation, inheritance, property decorators, polymorphism, duck typing, and object association structures (composition vs. aggregation).

### 9.1 OOP Basics & Class Mechanics

#### 1. Classes vs. Instances
* **Class (类)**: A blueprint or template for creating objects.
* **Instance (实例)**: A concrete object instantiated from a class (e.g. `kenwood = Kettle()`).

#### 2. The `self` Parameter
* In Python, all instance methods must explicitly list `self` as their first parameter.
* `self` represents the current instance of the class executing the method.
* **Implicit Argument Passing**: When calling `kenwood.switch_on()`, Python translates this under the hood to `Kettle.switch_on(kenwood)`. The object itself is implicitly passed as the first argument (`self`).

#### 3. Class Variables vs. Instance Variables
* **Instance Variables (实例变量)**: Variables defined inside the `__init__` constructor or other methods prefixed with `self.` (e.g. `self.price = 12.5`). They are stored in each instance's individual namespace (`__dict__`) and are unique to that object.
* **Class Variables (类变量)**: Variables declared directly inside the class body but outside any methods (e.g., `power_source = "electricity"`). They are shared by all instances of the class.
  * Accessing class variables: Can be accessed via the class (`Kettle.power_source`) or any instance (`kenwood.power_source`).

#### 4. Attribute Shadowing & Dynamic Attributes
* **Attribute Shadowing (属性遮蔽)**: If you assign a value to a class variable via an instance (e.g. `kenwood.power_source = "gas"`), Python does **not** change the class variable. Instead, it creates a new instance variable of the same name inside `kenwood`'s namespace. This new instance variable "shadows" (masks) the class variable for that specific instance.
* **Dynamic Attributes (动态属性)**: Python allows adding new attributes to an object dynamically at runtime (e.g. `kenwood.color = "red"`), even if that attribute was not declared in the class definition.

#### 5. String Representations: `__str__` vs. `__repr__`
Python provides two special magic methods (dunder methods) to customize how an object is converted to a string:
* **`__str__(self)`**:
  * **Intent**: Provide a readable, user-friendly string representation of the object.
  * **Audience**: End-users.
  * **Triggers**: Called by `print(obj)`, `str(obj)`, or formatting `f"{obj}"` / `"{}".format(obj)`.
* **`__repr__(self)`**:
  * **Intent**: Provide an unambiguous, detailed representation of the object, ideally showing the exact code needed to recreate the object (e.g. `eval(repr(obj)) == obj`).
  * **Audience**: Developers and debuggers.
  * **Triggers**: Called by `repr(obj)`, interactive shell display (typing the object name and pressing Enter), and when printing containers (e.g., printing a list of objects `[obj1, obj2]` calls `__repr__` for each item).
* **Fallback Behavior**:
  * If `__repr__` is defined but `__str__` is not, Python falls back to `__repr__` when printing the object.
  * If neither is defined, Python falls back to the default implementation inherited from `object`, which prints the class name and the memory address (e.g. `<__main__.Tag object at 0x...>`).

---

### 9.2 Encapsulation & Property Decorators

#### 1. Encapsulation & Attribute Privacy Rules
Python does not have strict access specifiers (like `private` or `protected` keywords in Java). Instead, it relies on naming conventions to signal privacy:
* **Public Attributes**: Names without leading underscores (e.g. `name`, `price`). Can be read/written from anywhere.
* **Protected/Non-Public Attributes (`_name`)**: Names starting with a single underscore (e.g. `_lives`).
  * **Rule**: This is a strong visual convention warning developers that the attribute is implementation-specific and should not be accessed directly outside the class. However, Python does not enforce this restriction at runtime.
* **Private Attributes (`__name`)**: Names starting with a double underscore (e.g. `__hit_points`).
  * **Rule**: Python enforces this using **Name Mangling (名称改写)**. The attribute is renamed to `_ClassName__attributeName` internally (e.g. `_Enemy__hit_points`) to prevent accidental access or name clashes in subclasses.

#### 2. Getters & Setters
* Used to wrap attributes so that reading and writing them goes through validation or translation functions rather than direct attribute access.

#### 3. Property Decorators (`@property` & `@name.setter`)
Python provides a modern, clean way to implement getters and setters using property decorators, allowing clients to access attributes using standard dot notation (`obj.name`) while running method logic under the hood.

* **Example syntax**:
  ```python
  class Enemy:
      def __init__(self, name, hit_points):
          self._name = name
          self._hit_points = hit_points

      @property
      def hit_points(self):
          """Getter method for hit_points."""
          return self._hit_points

      @hit_points.setter
      def hit_points(self, value):
          """Setter method for hit_points with validation."""
          if value < 0:
              self._hit_points = 0
          else:
              self._hit_points = value
  ```
* **How it works**:
  * Reading `enemy.hit_points` automatically executes the getter method decorated with `@property`.
  * Assigning `enemy.hit_points = 100` automatically executes the setter method decorated with `@hit_points.setter`.

---

### 9.3 Inheritance & Method Overriding

#### 1. Basic Inheritance & Subclassing
Inheritance allows a child class (subclass) to inherit attributes and methods from a parent class (superclass), modeling an "is-a" relationship.
* **Syntax**: `class Troll(Enemy):`
* **Object Base Class**: In Python 3, writing `class Enemy:` implicitly inherits from the built-in `object` base class (i.e. `class Enemy(object):`), which defines default dunder methods (`__str__`, `__repr__`, `__init__`, etc.).

#### 2. Initializing Parent State via `super()`
A subclass must initialize the base class attributes. This is done by calling `super().__init__(...)`.
* **Example**:
  ```python
  class Troll(Enemy):
      def __init__(self, name):
          super().__init__(name=name, lives=1, hit_points=23)
  ```
  `super()` dynamically resolves and calls the parent class's `__init__` method, ensuring base attributes are set up properly.

#### 3. Method Overriding (方法重写)
A subclass can redefine a method inherited from the parent class to specialize its behavior.
* **Overriding Example**:
  In `enemy.py`, `Vampyre` overrides `take_damage`:
  ```python
  class Vampyre(Enemy):
      def take_damage(self, damage):
          if not self.dodges():
              super().take_damage(damage=damage)  # Delegation to parent
  ```
  If `Vampyre` dodges, it skips taking damage; otherwise, it delegates to `Enemy.take_damage()` using `super()`.

---

### 9.4 Polymorphism & Duck Typing (鸭子类型)

#### 1. What is Polymorphism?
Polymorphism ("having many forms") is the ability to treat different objects in a uniform way using a shared interface or method signatures.

#### 2. Duck Typing (鸭子类型)
In dynamic languages like Python, polymorphism is realized via **Duck Typing**:
> *"If it walks like a duck and quacks like a duck, then it's a duck."*
* **Philosophy**: Python does not check class types or inheritance lines at runtime before calling a method. It only checks if the object supports the requested method signature at the moment of invocation.
* **Example from ducks.py**:
  ```python
  def test_duck(duck):
      duck.walk()
      duck.swim()
      duck.quack()
  ```
  Even though `Penguin` does not inherit from `Duck`, you can pass a `Penguin` instance to `test_duck()` because it implements `walk()`, `swim()`, and `quack()`.
* **Why it's useful**: Avoids rigid inheritance hierarchies. For instance, a `ToyDuck` class (which is a toy, not a biological bird) can be passed to `test_duck` as long as it supports those three methods.
* **Static Duck Typing (Protocols)**: Using `typing.Protocol`, Python allows developers to document and statically verify these behavioral interfaces at development time (via IDEs/mypy) without forcing subclassing.

---

### 9.5 Object Association Structures: Composition vs. Aggregation

Both composition and aggregation represent "has-a" relationships where an object is made up of other objects. However, they differ in object lifecycle ownership.

#### 1. Composition (组合 - Strong Ownership)
* **Definition**: The container class instantiates its sub-objects inside its own constructor (`__init__`).
* **Ownership**: The container class completely owns and manages the lifecycles of its components. If the container object is destroyed, all its component objects are destroyed with it ("同生共死").
* **Example in `HtmlDoc` (Composition)**:
  ```python
  class HtmlDoc(object):
      def __init__(self, title=None):
          self._doc_type = DocType()  # Created internally
          self._head = Head(title)    # Created internally
          self._body = Body()         # Created internally
  ```

#### 2. Aggregation (聚合 - Weak Ownership)
* **Definition**: The container class receives pre-existing instances of the sub-objects as arguments through its constructor.
* **Ownership**: The container class does not own the sub-objects. If the container object is destroyed, the sub-objects continue to exist independently ("聚沙成塔，散沙犹在").
* **Example in `HtmlDoc` (Aggregation)**:
  ```python
  class HtmlDoc(object):
      def __init__(self, doc_type, head, body):
          self._doc_type = doc_type  # Injected from outside
          self._head = head          # Injected from outside
          self._body = body          # Injected from outside
  ```
  Usage:
  ```python
  new_body = Body()
  new_docType = DocType()
  new_header = Head('Aggregation document')
  my_page = HtmlDoc(new_docType, new_header, new_body)
  # If my_page is deleted, new_body, new_docType, and new_header still exist.
  ```
