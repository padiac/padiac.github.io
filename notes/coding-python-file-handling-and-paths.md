This document compiles file input/output (I/O) methods, path management, context managers, and high-level structured data serialization (JSON & CSV) in Python.

---

## 1. File Path Resolution: Working Directory vs. Script Location

When opening files using relative paths (e.g., `open('data.txt')`), Python resolves the path based on the **Current Working Directory (CWD)** where the terminal command was executed, not where the script resides.

### 1.1 The CWD Issue
If a script is run from a parent folder, a simple relative path will fail with `FileNotFoundError`.
* **IDE Behavior**: IDEs (like PyCharm/VS Code) automatically change the CWD to the script's directory when clicking "Run", which masks this path issue.

### 1.2 Robust Script-Relative Path Resolution
To make scripts run successfully from any directory, dynamically compute the path relative to the script itself using `__file__` and the `os` module:
```python
import os

# Get absolute path to the directory containing the current script
script_dir = os.path.dirname(__file__)
# Construct the absolute path to the target file
file_path = os.path.join(script_dir, 'data.txt')

with open(file_path, 'r') as file:
    ...
```

---

## 2. File Reading Strategies

Python provides multiple methods to read text files, each with distinct memory and control-flow profiles.

### 2.1 File Object Iteration (`for line in file:`) - *Recommended*
Files in Python are **iterables**. You can loop over them directly:
```python
with open('poem.txt', 'r') as poem:
    for line in poem:
        print(line.rstrip())
```
* **Memory**: Extremely efficient. It reads line-by-line (lazy loading), keeping only one line in memory at any time. Ideal for files of any size (even gigabytes).
* **Safety**: Automatically detects EOF (End of File) and terminates cleanly.

### 2.2 `readline()` in a Loop
Reads a single line from the file. If called repeatedly (e.g., in a `while True:` loop), it returns the next line.
* **Dead Loop Risk**: At EOF, `readline()` returns an empty string `""`. If the loop does not explicitly check and break on `not line`, it will loop infinitely.
```python
# Risky without EOF check:
with open('poem.txt') as poem:
    while True:
        line = poem.readline().rstrip()
        print(line)  # Will spin infinitely at EOF!
```

### 2.3 `readlines()`
Reads the **entire** file at once and returns a list of strings (one string per line).
* **Use Case**: Good when you need list indexing/slicing (e.g., `lines[-1]` for last line) or reverse traversal (`reversed(lines)`).
* **Memory**: High. Loads the entire file content into memory. Do not use on large files.

### 2.4 `read()`
Reads the **entire** file into a single, massive string.
* **Use Case**: Good for regex matching across the entire document or character-level manipulation.
* **Memory**: High. Loads the entire file content into memory.

---

## 3. The `with` Statement (Context Manager)

Opening files manually (e.g., `file = open(...)`) requires calling `file.close()` to release the system resource.

### 3.1 The Danger of Manual Closing
If an exception occurs between `open()` and `close()`, the script crashes before closing the file, leading to resource leaks.

### 3.2 The `with` Context Manager - *Best Practice*
```python
with open('file.txt', 'r') as file:
    content = file.read()
# File is automatically closed here, even if exceptions occurred inside
```
* **How it works**: The context manager guarantees that the file is closed as soon as the execution leaves the `with` block, regardless of errors. Manual `close()` is never needed.

---

## 4. Writing to Files: `print()` vs. `.write()`

When opening a file in Write (`"w"`) mode, Python truncates (empties) the file if it exists, or creates it if it doesn't.

### 4.1 Comparison of Writing Methods

| Feature | `print(data, file=file_obj)` | `file_obj.write(string)` |
| :--- | :--- | :--- |
| **Formatting** | Automatic type conversion to string (calls `__str__()`). | Strictly requires string inputs (raises `TypeError` on ints). |
| **Newlines** | Automatically appends `\n` at the end (like terminal print). | Does not append newlines. Write exactly the input string. |

```python
# print() is highly convenient:
print(42, file=output_file)  # Writes '42\n'

# .write() requires manual conversion:
output_file.write(str(42) + '\n')  # Writes '42\n'
```

### 4.2 File Access Modes

When using the `open(filename, mode)` built-in function, the second argument dictates how Python accesses the system file:

| Mode | Name | File Existence Behavior | Existing Content Behavior | File Pointer Location |
| :--- | :--- | :--- | :--- | :--- |
| **`"r"`** | **Read** (Default) | Raises `FileNotFoundError` if missing. | Unchanged (Read-only). | Beginning of the file. |
| **`"w"`** | **Write** | Creates a new file if missing. | **Truncates** (Empties all content to 0 bytes). | Beginning of the file. |
| **`"a"`** | **Append** | Creates a new file if missing. | Unchanged (Appends new data). | End of the file. |

#### Binary Modes (`"b"` suffix)
For non-text files (such as images, audio, or compiled binaries), text decoding must be bypassed:
* Use `"rb"` (read binary) or `"wb"` (write binary).
* Reads and writes raw `bytes` objects instead of `str` objects. Encodings (like `'utf-8'`) are not permitted when using binary modes.

---

## 5. String Manipulation: Stripping & Sequence Unpacking

### 5.1 Trailing Whitespace: `rstrip()`
When reading a file line-by-line, the returned string `line` ends with a newline character (`\n`). Python's `print()` function also appends a newline by default.
* **Double Newline Bug**: Calling `print(line)` outputs a blank line between every text row.
* **Solution**: Call `line.rstrip()` to strip trailing whitespace and newlines before printing.

### 5.2 Splitting and Unpacking File Rows
If files contain structured data (e.g. pipe-delimited `country_info.txt`: `Name|Capital|Code`), split the row and unpack it:
```python
# row.strip('\n').split('|') returns a list: ['France', 'Paris', 'FR']
country, capital, code = row.strip('\n').split('|')  # Unpacks list into variables
```
* **Constraint**: The number of variables on the left must exactly match the number of elements returned by `.split()` on the right, otherwise Python raises a `ValueError`.

---

## 6. JSON Serialization: Mapping & Limitations

JSON (JavaScript Object Notation) is a lightweight, text-based data format commonly used to serialize and share nested Python data structures.

### 6.1 Python to JSON Type Mapping
Python's `json` module recursively serializes Python objects to JSON types according to this standard mapping:

| Python Type | JSON Type | JSON Example |
| :--- | :--- | :--- |
| `dict` (字典/哈希表) | `object` (对象) | `{"name": "Bob", "age": 30}` *(Keys must be double-quoted strings)* |
| `list`, `tuple` | `array` (数组) | `["ABC", 1987]` *(Square brackets)* |
| `str` (字符串) | `string` (字符串) | `"Hello"` *(Double quotes strictly required)* |
| `int`, `float` | `number` (数字) | `1987` or `3.14` |
| `bool` | `boolean` (布尔值) | `true` or `false` *(Lowercased)* |
| `None` | `null` | `null` |

### 6.2 Method Naming: The `s` Suffix (String vs. File)
The `json` module provides two pairs of methods. The trailing **`s`** stands for **String**:

* **Reading (Deserialization)**:
  * `json.load(fp)`: Reads and parses JSON directly from a **file object** `fp`.
  * `json.loads(s)`: Parses JSON directly from a **string object** `s` in memory.
* **Writing (Serialization)**:
  * `json.dump(obj, fp)`: Serializes `obj` and writes it directly to a **file object** `fp`.
  * `json.dumps(obj)`: Serializes `obj` and returns it as a **string** in memory.

### 6.3 Limitations of JSON
* **Loss of Type Specificity**: JSON does not support custom/complex types (e.g. complex numbers, datetimes, sets).
  * *Example*: A complex number `12.5 + 3i` must be serialized as a generic list `[12.5, 3]`. The receiving program must manually know to reconstruct it.
* **The Tuple-to-List Loss (不可逆性)**: 
  * Python `tuple` objects are serialized as JSON arrays `[...]`.
  * When read back using `json.load()`, Python cannot tell if the array was originally a list or a tuple, so **it restores it as a mutable `list`**. Immutability is lost.

---

## 7. CSV (Comma-Separated Values) Handling

CSV files represent tabular data where columns are separated by commas. Python handles this format using the built-in `csv` module.

### 7.1 Basic Reading and the `newline=''` Rule
When opening CSV files, always specify `newline=''` in `open()`:
```python
with open('data.csv', encoding='utf-8', newline='') as csv_file:
    reader = csv.reader(csv_file)
```
* **Why**: Prevents differences in cross-platform newline handling (e.g., `\r\n` vs `\n`) from corrupting the parsing behavior of the `csv` reader/writer.

### 7.2 Automatic Numeric Parsing: `QUOTE_NONNUMERIC`
By default, the `csv` reader treats all read values as strings. Specify `quoting=csv.QUOTE_NONNUMERIC` to automatically parse numeric columns:
* **Behavior**: Any value **not** enclosed in double quotes `""` is automatically converted to a float. Values in quotes remain strings.
```python
reader = csv.reader(csv_file, quoting=csv.QUOTE_NONNUMERIC)
```

### 7.3 Header (标题行/表头) Handling and Crashes
Many CSV files contain a header row. If we use `quoting=csv.QUOTE_NONNUMERIC`, the header row's formatting is critical:
* **Unquoted Headers (e.g., `Rank,Country...`)**:
  * Since there are no quotes, `QUOTE_NONNUMERIC` assumes `Rank` is a number and tries to cast it to a float.
  * **Result**: Raises a `ValueError: could not convert string to float: 'Rank'` and crashes the program.
  * **Fix**: Use `readline()` to consume and skip the first line before passing the file to `csv.reader`:
    ```python
    # Read, split, and skip the header row manually
    headers = csv_file.readline().strip('\n').split(',')
    reader = csv.reader(csv_file, quoting=csv.QUOTE_NONNUMERIC)
    ```
* **Quoted Headers (e.g., `"Cereal","Calories"...`)**:
  * Since they have double quotes, `QUOTE_NONNUMERIC` recognizes them as strings and does not cast them, so no crash occurs.

### 7.4 `csv.DictReader`: Reading CSVs into Dictionaries
Unlike `csv.reader` (which returns a `list` of strings/numbers for each row), `csv.DictReader` parses each row into a **Python Dictionary (`dict`)**.
* **Key-Value Mapping**: It maps column names (from the header row) as keys to the corresponding fields as values.
* **Lazy Loading / Iterator**: `csv.DictReader` is an **iterator**. When looping over it with `for row in reader:`, it processes the file **line-by-line**, keeping only one dictionary representing the current row in memory at any time.
* **Storing All Rows**: To load the entire file into a single structure in memory, cast the reader to a list: `all_rows = list(reader)`. This yields a **list of dictionaries** (字典列表):
  ```python
  [
      {'Cereal': 'Barley', 'Calories': '556', ...},
      {'Cereal': 'Durum', 'Calories': '339', ...}
  ]
  ```

### 7.5 Auto-Detecting CSV Layout: `csv.Sniffer`
When the separator (comma, tab, semicolon) or quoting rules of a CSV-like file are unknown, use `csv.Sniffer` to auto-detect the formatting details (the **dialect**):
```python
with open('data.txt', encoding='utf-8', newline='') as csv_file:
    # 1. Read a representative sample of the file (e.g. first 3 lines)
    sample = ""
    for _ in range(3):
        sample += csv_file.readline()
    
    # 2. Sniff the dialect from the sample
    detected_dialect = csv.Sniffer().sniff(sample)
    
    # 3. Rewind the file pointer to the beginning
    csv_file.seek(0)
    
    # 4. Read the file using the sniffed dialect
    reader = csv.reader(csv_file, dialect=detected_dialect)
```
> [!IMPORTANT]
> **File Pointer Rewinding (`seek(0)`)**: Reading the sample advances the file pointer. You **must** call `file.seek(0)` to rewind to the beginning before passing the file object to the CSV reader, otherwise the first lines of the file will be skipped.

### 7.6 Customizing Dialects: `csv.excel` Templates
Alternatively, you can manually define or customize a CSV formatting dialect by modifying properties of `csv.excel` (the template matching standard Excel formatting):
```python
# Create or grab the excel template
dialect = csv.excel
# Customize the separator character to use pipe '|'
dialect.delimiter = '|'
# Turn on stripping of leading spaces after delimiters
dialect.skipinitialspace = True

with open('data.txt', encoding='utf-8', newline='') as file:
    reader = csv.DictReader(file, dialect=dialect)
```

### 7.7 Resilient Dict Keys: Case-Insensitive Header Normalization
By default, the keys of dictionaries returned by `csv.DictReader` strictly match the spelling and capitalization of the header row in the file. If the file's header capitalization changes unexpectedly, code accessing keys (like `row['Country']`) will raise `KeyError` and crash.

To build robust programs, read the headers manually, normalize them (e.g., lowercase them using `.casefold()`), and provide them to `DictReader` via `fieldnames`:
```python
with open('data.txt', encoding='utf-8', newline='') as file:
    # 1. Manually read and strip the header row
    raw_header = file.readline().strip('\n').split('|')
    
    # 2. Normalize headers to lowercase
    normalized_headers = [h.casefold() for h in raw_header]
    
    # 3. Feed DictReader with normalized headers
    # Note: DictReader will start reading from the current pointer (line 2), avoiding duplicate header reading.
    reader = csv.DictReader(file, dialect=dialect, fieldnames=normalized_headers)
    for row in reader:
        # Accessing keys is now safe and case-insensitive
        print(row['country'], row['capital'])
```

### 7.8 Aliasing / Double-Indexing Dictionary Entries
When indexing datasets for lookup (such as looking up capital cities by country name), you can map the same record dictionary to **multiple keys** in your lookup table. This allows users to query by either the country name or the country code:
```python
countries_db = {}
for row in reader:
    # Index by full country name (lowercased)
    countries_db[row['country'].casefold()] = row
    # Index by country code (lowercased)
    countries_db[row['cc'].casefold()] = row
```
* **Effect**: `countries_db['andorra']` and `countries_db['ad']` point to the exact same dictionary object in memory, providing convenient aliases for lookup.

---

## 8. Directory Traversal and Path Manipulation (`os.walk` & `os.path.split`)

When processing collections of files (like a music library structured by Artist/Album/Songs), Python provides standard utility functions in the `os` module to traverse the directories and split paths.

### 8.1 Directory Tree Traversal with `os.walk()`
`os.walk(root, topdown=True)` is a generator that recursively traverses a directory tree. For every directory it visits, it yields a 3-element tuple:
`(path, directories, files)`

* **`path` (str)**: The string path to the current directory being visited.
* **`directories` (list)**: A list of names of subdirectories inside the current directory.
* **`files` (list)**: A list of filenames inside the current directory.

#### How the traversal behaves step-by-step:
Suppose we have the path structure: `music/AC DC/For Those About To Rock (We Salute You)/1 - Song.emp3`.
1. **Level 1 (Root)**: `os.walk` visits `music/`.
   - `path = "music"`
   - `directories = ["AC DC", "Beatles", ...]`
   - `files = []`
2. **Level 2 (Artist)**: `os.walk` visits `music/AC DC/`.
   - `path = "music/AC DC"`
   - `directories = ["For Those About To Rock (We Salute You)", ...]`
   - `files = []`
3. **Level 3 (Album)**: `os.walk` visits `music/AC DC/For Those About To Rock (We Salute You)/`.
   - `path = "music/AC DC/For Those About To Rock (We Salute You)"`
   - `directories = []`
   - `files = ["1 - For Those About To Rock (We Salute You).emp3", ...]`
   *At this point, we have reached the folder containing the actual files.*

### 8.2 Path Splitting with `os.path.split()`
`os.path.split(path)` splits a path string into two parts at the **last** slash `/`:
1. **Head (Parent Path)**: Everything leading up to the final directory or file.
2. **Tail (Final Name)**: The final directory or file name.

#### Double Splitting Example:
If `path` is `"music/AC DC/For Those About To Rock (We Salute You)"`:
```python
# First split yields parent directory and album name:
first_split = os.path.split(path)
# Returns: ('music/AC DC', 'For Those About To Rock (We Salute You)')
# Album Name is first_split[1]

# Second split on the parent yields root and artist name:
second_split = os.path.split(first_split[0])
# Returns: ('music', 'AC DC')
# Artist Name is second_split[1]
```

### 8.3 File Extension Stripping & Splitting
To parse metadata directly from filenames (e.g., `"1 - Song Name.emp3"`):
1. **Strip Extension via Negative Slicing (`[:-5]`)**:
   - The extension `.emp3` has exactly 5 characters.
   - Slicing `f[:-5]` returns the string from the beginning up to the last 5 characters, stripping `.emp3`.
   - Example: `"1 - Song.emp3"[:-5]` $\rightarrow$ `"1 - Song"`.
2. **Split on Delimiter (`.split(' - ')`)**:
   - Calling `.split(' - ')` splits the string by the separator `' - '` (space, dash, space).
   - Example: `"1 - Song".split(' - ')` $\rightarrow$ `['1', 'Song']`.
   - The first element is the track number, and the second is the song title.
