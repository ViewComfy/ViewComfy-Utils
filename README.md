# ViewComfy Utils

Custom ComfyUI nodes to enable ViewComfy app functionalities like workflow branching, optional image inputs, input validation and custom errors. 

## Nodes

1. [Compare](#compare)

2. [Conditional Select](#conditional-select)

3. [Show Anything](#show-anything)

4. [Load Image](#load-image)

5. [Anything Inversed Switch](#anything-inversed-switch)

6. [Show Error Message](#show-error-message)

  

---

  

## Compare

Compares two inputs and returns a Boolean. 

### Inputs
Supports multiple data types, including primitives, tensors (images or videos), Booleans, etc. 

The default value for a and b is `None`

### Supported operators

-  `a == b`

-  `a != b`

-  `a < b`

-  `a > b` 

-  `a <= b` 

-  `a >= b`

-  `a and b`

-  `a or b` 

  
### Outputs

-  Boolean

### Special Behavior

- Only `==` and `!=` operators are supported for Tensor Comparison (e.g. Images and Videos)

-  The `and` and `or` operations require both inputs to be boolean values.

  

### Example Use Cases

- Comparing numeric values to control workflow branching

- Checking if tensors/images are identical

- Create custom input validations in ComfyUI workflows 

---

  

## Conditional Select

Outputs one of two input values based on a boolean condition. This is essentially a programmatic if-else statement that works with any data type.

  

### Inputs

- A boolean value that determines which input to output. Default: `True`

- Values A and B. They can be of any type. Default: `None`


### Outputs

-  Returns Value A if the condition is True, otherwise returns Value B.

### Example Use Cases

- Selecting between different models, images, or parameters based on conditions.

- Implementing conditional logic in ComfyUI workflows. 

  

---

  

## Show Anything

A visualization node that displays any type of data in the ComfyUI interface. It can handle strings, lists, objects, and complex data structures by converting them to readable text format.


### Inputs

-  Anything


### Outputs

-  The input as a JSON string
  

### Example Use Cases

- Can be used to display text in ViewComfy apps
- Debugging workflows by inspecting intermediate values
- Monitoring data flow through complex workflows

 
---

  

## Load Image

A variation of the Load Image node with optional image loading capability. This allows ComfyUI workflows to run even when the node is not pointing to any image.  

### Inputs

- An Image. Default: `"None"`

  

### Outputs

- An image and/or a mask
- Returns `None` if no image is selected or if the file doesn't exist

  

### Special Behavior

-  Unlike standard ComfyUI LoadImage, this node can work with `"None"` or `""` as input, returning None instead of raising an error.
  

### Example Use Cases

- Build ViewComfy apps or ComfyUI based APIs with optional image inputs.

  

---

  

## Anything Inversed Switch

Routes a single input to one of multiple outputs (up to 10) based on an index value. All non-selected outputs are blocked from execution, making this useful for conditional execution paths. 

### Inputs

- An index value to determine which output to route the input to. Default: `0`

- The value to route to the selected output. This can be of any data type.
  

### Outputs

- Only the output at the position matching `index` will receive the input value

- All other outputs are blocked. 

  

### Special Behavior

-  Nodes downstream of non-selected outputs will not execute
  

### Example Use Cases

- Creating ViewComfy apps with conditional execution paths based on a user selected index

- Routing data to different processing pipelines based on conditions

- Building switch/case-like logic in ComfyUI

  

---

  

## Show Error Message

Conditionally displays an error message and halts workflow execution. 

  

### Inputs

- A boolean to control whether to raise an error or not. Default: `True`

- The first and second parts of the error message. 


### Outputs

-  An error showing both parts of the messages combined. 
  

### Example Use Cases

- Validating workflow inputs

- Creating conditional error messages based on workflow state

- Implementing user-friendly error handling in complex workflows

