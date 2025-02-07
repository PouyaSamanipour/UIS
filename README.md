# **Replacing $\mathcal{K}_\infty$ Function with Leaky ReLU in Barrier Function Design: A Union of Invariant Sets Approach for ReLU-Based Dynamical Systems**

This repository contains the implementation of a systematic framework for determining **piecewise affine (PWA) barrier functions** and their corresponding **invariant sets** for dynamical systems identified via **Rectified Linear Unit (ReLU) neural networks** or their equivalent **PWA representations**.

---

## **Abstract**
We present the **Union of Invariant Sets (UIS) method**, which aggregates information from multiple invariant sets to compute the **largest possible PWA invariant set**.  
This framework has been validated through various examples, including the **Inverted Pendulum** and **Double Integrator**, demonstrating its capability to improve the analysis of invariant sets in ReLU-based dynamical systems.  

For more details, refer to our paper on ArXiv: [https://arxiv.org/abs/2502.03765](https://arxiv.org/abs/2502.03765).

---

## **Installation and Requirements**

### **1. Clone the Repository**
```bash
git clone https://github.com/PouyaSamanipour/UIS.git
cd UIS
```

### **2. Set Up a Python Virtual Environment**
Create a virtual environment to avoid conflicts with system-wide packages:
```bash
python -m venv uis_env
uis_env\Scripts\activate
```

### **3. Install Dependencies**
Install the required dependencies using the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

### **4. Install Gurobi**
This project relies on **Gurobi** for solving optimization problems:
1. Download and install Gurobi from the [official website](https://www.gurobi.com/).
2. Obtain an **academic license** if eligible (free for academic users).
3. Set up your license following Gurobi's [license setup guide](https://www.gurobi.com/documentation/quickstart.html).

After installation, ensure the Gurobi Python interface is available in your environment:
```bash
pip install gurobipy
```

---

## **How to Run the Code**

### **1. Navigate to the Project Directory**
```bash
cd UIS
```

### **2. Run Example Scripts**
Example scripts are available in the `Examples` folder. Two examples are provided:
- **Inverted Pendulum**
To run a specific example, use:
```bash
python Examples/IP.py
```


### **3. Customize the Framework**
Modify the input parameters in the scripts to test different ReLU-based dynamical systems or PWA representations.


## **License**
This project is free for academic use under the MIT license. Please refer to the `LICENSE` file for more details.

---

## **Citation**
For more information, refer to our paper on ArXiv: [https://arxiv.org/abs/2502.03765](https://arxiv.org/abs/2502.03765).

---

## **Contact**
For questions or inquiries, please contact **Pouya Samanipour** at [psa254@uky.edu](mailto:psa254@uky.edu).

