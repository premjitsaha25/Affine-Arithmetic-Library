import numpy as np

from Affine_ArithmeticClassV3 import AffineArray

def scale(x, lo, hi):
    return lo + (hi - lo)*x

def test_Branin(x, cheb=False):

    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5  / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)

    x1 = x[0]
    x2 = x[1]

    x1 = 15 * x1 - 5
    x2 = 15 * x2

    funVal = (
        a * (x2 - b * x1.pow(2, cheb=cheb) + c * x1 - r).pow(2, cheb=cheb) 
        + s * (1 - t) * (x1).cos(cheb=cheb) + s + 5*x1
    )

    return funVal


def test_Ackley(x, d, cheb=False):

    a = 20.0
    b = 0.2
    c = 2*np.pi

    funVal = a + np.exp(1)

    sum1 = 0.0
    sum2 = 0.0

    for i in range(d):

        # xi = 2*32.768*x[i] - 32.768
        xi = scale(x[i], -32.768, 32.768)

        sum1 += xi.pow(2, cheb=False)
        sum2 += (c * xi).cos(cheb=cheb)

    term1 = (-b * (sum1/d).sqrt(cheb=cheb)).exp(cheb=cheb)
    term2 = ((sum2/d)).exp(cheb=cheb)

    funVal += -a * term1
    funVal += -term2

    return funVal

def test_Eggholder(x, cheb=False):

    # x1 = 512 * x[0] - 512
    # x2 = 512 * x[1] - 512

    x1 = scale(x[0], -512, 512)
    x2 = scale(x[1], -512, 512)

    funVal = (
        -(x2 + 47)
        * (
            (x2 + 47 + x1/2)
            .abs(cheb=cheb)
            .sqrt(cheb=cheb)
        ).sin(cheb=cheb)

        - x1
        * (
            (x1 - (x2 + 47))
            .abs(cheb=cheb)
            .sqrt(cheb=cheb)
        ).sin(cheb=cheb)
    )

    return funVal


X = AffineArray.from_intervals([
    (0, 1),
    (0, 1)
])

#------------------------------------------------
#  Tset Branin function
#------------------------------------------------

# Store results per split configuration
results_Branin = {}

# maximum number of splits, 2^dim1 in each direction
dim1 = 9

print("======================================= \n"
    f"Results from Branin function \n"
    "=======================================")

for m in range(dim1):

    splits_per_dim = [2**m, 2**m]

    subintervals = X.split_interval(splits_per_dim)

    # Reset accumulators for each (m, n) configuration          
    lo_Ch, hi_Ch = np.inf, -np.inf
    lo_MR, hi_MR = np.inf, -np.inf

    for Xi in subintervals:

        # --- Min-range fit ---

        funVal_MR = test_Branin(Xi,cheb=False)

        frange_MR = funVal_MR.interval

        lo_MR = min(lo_MR, frange_MR[0])
        hi_MR = max(hi_MR, frange_MR[1])

        # --- Chebyshev fit ---
        
        funVal_ch = test_Branin(Xi, cheb=True)

        frange_ch = funVal_ch.interval

        lo_Ch = min(lo_Ch, frange_ch[0])
        hi_Ch = max(hi_Ch, frange_ch[1])

    # Store result for this (m, n) configuration
    results_Branin[(2**m, 2**m)] = {
        'f_rangeCh' : [lo_Ch, hi_Ch],
        'f_rangeMR' : [lo_MR, hi_MR],
        'f_intersect': [max(lo_Ch, lo_MR), min(hi_Ch, hi_MR)]
    }

    print(f"splits={splits_per_dim}  |  "
        f"Chebyshev: [{lo_Ch:.6f}, {hi_Ch:.6f}]  |  "
        f"Min-range: [{lo_MR:.6f}, {hi_MR:.6f}]  |  "
        f"f_intersect: [{max(lo_Ch, lo_MR):.4f}, {min(hi_Ch, hi_MR):.4f}]")


#------------------------------------------------
#  Tset Egg-holder function 
#------------------------------------------------

# Store results per split configuration
results_Eggholder = {}

# maximum number of splits, 2^dim1 in each direction
dim1 = 9

print("======================================= \n"
    f"Results from Egg-holder function \n"
    "=======================================")

for m in range(dim1):

    splits_per_dim = [2**m, 2**m]

    subintervals = X.split_interval(splits_per_dim)

    # Reset accumulators for each (m, n) configuration          
    lo_Ch, hi_Ch = np.inf, -np.inf
    lo_MR, hi_MR = np.inf, -np.inf

    for Xi in subintervals:

        # --- Min-range fit ---

        funVal_MR = test_Eggholder(Xi,cheb=False)

        frange_MR = funVal_MR.interval

        lo_MR = min(lo_MR, frange_MR[0])
        hi_MR = max(hi_MR, frange_MR[1])

        # --- Chebyshev fit ---
        
        funVal_ch = test_Eggholder(Xi, cheb=True)

        frange_ch = funVal_ch.interval

        lo_Ch = min(lo_Ch, frange_ch[0])
        hi_Ch = max(hi_Ch, frange_ch[1])

    # Store result for this (m, n) configuration
    results_Branin[(2**m, 2**m)] = {
        'f_rangeCh' : [lo_Ch, hi_Ch],
        'f_rangeMR' : [lo_MR, hi_MR],
        'f_intersect': [max(lo_Ch, lo_MR), min(hi_Ch, hi_MR)]
    }

    print(f"splits={splits_per_dim}  |  "
        f"Chebyshev: [{lo_Ch:.6f}, {hi_Ch:.6f}]  |  "
        f"Min-range: [{lo_MR:.6f}, {hi_MR:.6f}]  |  "
        f"f_intersect: [{max(lo_Ch, lo_MR):.4f}, {min(hi_Ch, hi_MR):.4f}]")

#------------------------------------------------
#  Tset Ackley function 2-D case
#------------------------------------------------

# Store results per split configuration
results_Ackley2D = {}

# maximum number of splits, 2^dim1 in each direction
dim1 = 9

print("======================================= \n"
    f"Results from Ackley function 2-D case \n"
    "=======================================")

for m in range(dim1):

    splits_per_dim = [2**m, 2**m]
    
    subintervals = X.split_interval(splits_per_dim)

    # Reset accumulators for each (m, n) configuration          
    lo_Ch, hi_Ch = np.inf, -np.inf
    lo_MR, hi_MR = np.inf, -np.inf

    for Xi in subintervals:

        # --- Min-range fit ---
        funVal_MR = test_Ackley(Xi, d=2, cheb=False)

        frange_MR = funVal_MR.interval

        lo_MR = min(lo_MR, frange_MR[0])
        hi_MR = max(hi_MR, frange_MR[1])

        # --- Chebyshev fit ---
        
        funVal_ch = test_Ackley(Xi, d=2, cheb=True)

        frange_ch = funVal_ch.interval

        lo_Ch = min(lo_Ch, frange_ch[0])
        hi_Ch = max(hi_Ch, frange_ch[1])

    # Store result for this (m, n) configuration
    results_Branin[(2**m, 2**m)] = {
        'f_rangeCh' : [lo_Ch, hi_Ch],
        'f_rangeMR' : [lo_MR, hi_MR],
        'f_intersect': [max(lo_Ch, lo_MR), min(hi_Ch, hi_MR)]
    }

    print(f"splits={splits_per_dim}  |  "
        f"Chebyshev: [{lo_Ch:.6f}, {hi_Ch:.6f}]  |  "
        f"Min-range: [{lo_MR:.6f}, {hi_MR:.6f}]  |  "
        f"f_intersect: [{max(lo_Ch, lo_MR):.4f}, {min(hi_Ch, hi_MR):.4f}]")
    


#------------------------------------------------
#  Tset Ackley function 3-D case
#------------------------------------------------

X = AffineArray.from_intervals([
    (0, 1),
    (0, 1),
    (0, 1)
])

# Store results per split configuration
results_Ackley3D = {}

# maximum number of splits, 2^dim1 in each direction
dim1 = 6

print("======================================= \n"
    f"Results from Ackley function 3-D case \n"
    "=======================================")

for m in range(dim1):

    splits_per_dim = [2**m, 2**m, 2**m]

    subintervals = X.split_interval(splits_per_dim)

    # Reset accumulators for each (m, n) configuration          
    lo_Ch, hi_Ch = np.inf, -np.inf
    lo_MR, hi_MR = np.inf, -np.inf

    for Xi in subintervals:

        # --- Min-range fit ---

        funVal_MR = test_Ackley(Xi, d=3, cheb=False)

        frange_MR = funVal_MR.interval

        lo_MR = min(lo_MR, frange_MR[0])
        hi_MR = max(hi_MR, frange_MR[1])

        # --- Chebyshev fit ---

        funVal_ch = test_Ackley(Xi, d=3, cheb=True)

        frange_ch = funVal_ch.interval

        lo_Ch = min(lo_Ch, frange_ch[0])
        hi_Ch = max(hi_Ch, frange_ch[1])

    # Store result for this (m, n) configuration
    results_Branin[(2**m, 2**m, 2**m)] = {
        'f_rangeCh' : [lo_Ch, hi_Ch],
        'f_rangeMR' : [lo_MR, hi_MR],
        'f_intersect': [max(lo_Ch, lo_MR), min(hi_Ch, hi_MR)]
    }

    print(f"splits={splits_per_dim}  |  "
        f"Chebyshev: [{lo_Ch:.6f}, {hi_Ch:.6f}]  |  "
        f"Min-range: [{lo_MR:.6f}, {hi_MR:.6f}]  |  "
        f"f_intersect: [{max(lo_Ch, lo_MR):.4f}, {min(hi_Ch, hi_MR):.4f}]")