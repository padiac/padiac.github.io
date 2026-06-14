This document explores character encoding systems, their binary representations, and variable-length parsing logic in Python.

---

## 1. Introduction to Character Encodings (ASCII, EBCDIC, and Unicode)

Character encoding is the mapping system used to translate characters (human language) into binary bytes (computer language).

* **ASCII**: Old 7-bit standard. Can only represent 128 characters (English alphabet, numbers, basic symbols). Fails on any non-English script.
* **EBCDIC**: Old 8-bit IBM mainframe standard. Obsolete.
* **Unicode**: The universal standard. Maps **almost every character** in the world (all languages, math symbols, emoji) to a unique number called a **Code Point** (e.g. `U+4E2D` for `"中"`).

---

## 2. UTF-8 Encoding & Variable-Length Decoding

Unicode only maps characters to numbers; it doesn't specify how to store them. **UTF-8** is the encoding format that translates those numbers into bytes.

* **Variable Width**: UTF-8 uses 1 byte for English (backward compatible with ASCII), 3 bytes for Chinese/common scripts, and 4 bytes for emojis.
* **Efficiency**: Highly space-efficient and globally compatible. It is the default encoding for web pages and Python 3.
* **Others**:
  * **UTF-16**: Uses 2 or 4 bytes. Used in JVM/Windows memory. Space-inefficient for English.
  * **UTF-32**: Uses fixed 4 bytes. Simple lookup but massive storage waste.

### 2.1 Variable-Length Decoding: How UTF-8 Knows Where to Split Bytes

Since UTF-8 is a variable-length encoding (characters can be 1, 2, 3, or 4 bytes long), the decoder needs a reliable way to determine where one character ends and the next begins in a continuous byte stream. It achieves this using a **prefix bit-pattern marking system**:

| Leading Bits of Byte | Byte Type | Character Length / Meaning |
| :--- | :--- | :--- |
| **`0xxxxxxx`** | **Single Byte** | 1-byte character (ASCII). |
| **`110xxxxx`** | **Leading Byte** | Starts a 2-byte character. |
| **`1110xxxx`** | **Leading Byte** | Starts a 3-byte character. |
| **`11110xxx`** | **Leading Byte** | Starts a 4-byte character. |
| **`10xxxxxx`** | **Continuation Byte** | Follows a leading byte in multi-byte characters. |

#### Case Study: Decoding `207, 128, 114, 194, 178` (represented as `b'\xcf\x80r\xc2\xb2'`)

When Python decodes this 5-byte sequence using UTF-8, it looks at the binary representation of each byte sequentially:

1. **Byte 1: `207` (binary `11001111`)**:
   - Starts with `110...` $\rightarrow$ The decoder knows this is the start of a **2-byte character**. It will automatically combine this byte with the next byte.
2. **Byte 2: `128` (binary `10000000`)**:
   - Starts with `10...` $\rightarrow$ Valid continuation byte.
   - **Result**: `[207, 128]` are combined and decoded to the Greek letter **`π`**.
3. **Byte 3: `114` (binary `01110010`)**:
   - Starts with `0...` $\rightarrow$ The decoder knows this is a **1-byte character** (ASCII).
   - **Result**: Decoded directly as the character **`r`**.
4. **Byte 4: `194` (binary `11000010`)**:
   - Starts with `110...` $\rightarrow$ Start of a **2-byte character**.
5. **Byte 5: `178` (binary `10110010`)**:
   - Starts with `10...` $\rightarrow$ Valid continuation byte.
   - **Result**: `[194, 178]` are combined and decoded as the superscript **`²`**.

By using these binary header flags, the decoder can parse a continuous stream of bytes without any ambiguity or need for separate separators, returning exactly **`πr²`**.

---

## 3. Best Practices to Avoid Encodings/乱码 Issues
* **Gold Rule**: Always write and read files in UTF-8.
* **In Python**: Explicitly define the `encoding` parameter in `open()` calls:
  ```python
  with open('file.txt', 'r', encoding='utf-8') as f:
  ```
* **Legacy Files**: If reading old files encoded in regional systems (like `GBK` for Chinese, `CP1252` for West Europe), read them with `encoding='gbk'` or `encoding='cp1252'`, but **always save output files using `encoding='utf-8'`**.
