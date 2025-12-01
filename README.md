# Teensor ⚡

**Teensor** is a lightweight Python library for tensor operations with automatic differentiation (Autograd/Backpropagation) support.

## Requirements

- **NumPy** must be installed (`pip install numpy`) since Teensor is built on top of NumPy.  

## 🚀 Features

* **Automatic Differentiation:** Built-in backward engine to compute gradients automatically.
* **Tensor Operations:** Supports element-wise addition, multiplication, power.
* **Linear Algebra:** Matrix multiplication (`matmul`), Transpose, and Inverse.
* **Numpy Integration:** Built on top of NumPy for efficient computation.

---

## 📦 Installation

Since this is a local package, ensure the `teensor` directory is in your project root.

```python
# Import directly from the package
from teensor import Teensor
```

---

## 🔹 Usage Examples

### 1. Creating Tensors
You can create tensors from lists or NumPy arrays.

```python
from teensor import Teensor

a = Teensor([[1.0, 2.0], 
             [3.0, 4.0]])

b = Teensor([[5.0, 6.0], 
             [7.0, 8.0]])

print(a)
# Output: Teensor(Data:[[1. 2.] [3. 4.]], dtype:Teensor)
```

### 2. Element-wise Operations
Standard mathematical operations work element-wise.

```python
# Addition
c = a + b

# Multiplication (Hadamard product)
d = a * b

# Power
e = a ** 2
```

### 3. Linear Algebra Operations
Teensor supports essential linear algebra operations.

```python
# Matrix Multiplication (Dot Product)
f = a.matmul(b)

# Transpose (Property)
g = a.transpose

# Inverse (Property - Only for square 2D matrices)
h = a.inverse
```

---

## 🧠 Backpropagation (Autograd)

Teensor tracks operations to build a computational graph. Call `.backward()` on the final scalar (or tensor) to compute gradients.

```python
# 1. Define Tensors
x = Teensor([[1.0, 2.0], [3.0, 4.0]])
y = Teensor([[1.0, 0.0], [0.0, 1.0]])

# 2. Forward Pass
# Operations: (x + y) -> matmul(x)
z = (x + y).matmul(x) 
loss = z.sum()

# 3. Backward Pass
loss.backward()

# 4. Access Gradients
print("Gradients of x:\n", x.grad)
print("Gradients of y:\n", y.grad)
```

---

## 📝 Operation Details

* **Element-wise Ops:** Operators like `+`, `*`, and `**` are element-wise.
* **Linear Algebra:** Use `.matmul()`, `.transpose`, and `.inverse` for matrix operations.
