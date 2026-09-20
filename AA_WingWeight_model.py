import numpy as np

from Affine_ArithmeticClassV3 import AffineArray

def scale(x, lo, hi):
    return lo + (hi - lo)*x

def WingWeight_model(x, cheb=False):

    Sw = scale(x[0], 150, 200)
    Wf = scale(x[1], 220, 300)
    Area = scale(x[2], 6, 10)
    Lamb = scale(x[3], -10*np.pi/180, 10*np.pi/180)
    q = scale(x[4], 16, 45)
    lamb = scale(x[5], 0.5, 1)
    tc = scale(x[6], 0.08, 0.18)
    Nz = scale(x[7], 2.5, 6)
    Wdg = scale(x[8], 1700, 2500)
    Wp = scale(x[9], 0.025, 0.08)

    # W_fun =( 0.036*(Sw.pow(0.758, cheb=cheb))*Wf.pow(0.0035, cheb=cheb) * 
    #     (Area/(Lamb.cos(cheb=cheb).pow(2, cheb=cheb))).pow(0.6, cheb=cheb) * 
    #     q.pow(0.006, cheb=cheb) *
    #     lamb.pow(0.04, cheb=cheb) * 
    #     (100 * tc/(Lamb.cos(cheb=cheb))).pow(-0.3, cheb=cheb) *
    #     (Nz * Wdg).pow(0.49, cheb=cheb) 
    #     + Sw * Wp
    # )

    # para1 = 0.036*(Sw.pow(0.758, cheb=cheb))*Wf.pow(0.0035, cheb=cheb)
    # para2 = (Area/(Lamb.cos(cheb=cheb).pow(2, cheb=cheb))).pow(0.6, cheb=cheb)
    # para3 = q.pow(0.006, cheb=cheb)
    # para4 = lamb.pow(0.04, cheb=cheb) 
    # para5 = (100 * tc/(Lamb.cos(cheb=cheb))).pow(-0.3, cheb=cheb)
    # para6 = (Nz * Wdg).pow(0.49, cheb=cheb)
    # para7 = Sw * Wp

    # para8 = Area/(Lamb.cos(cheb=cheb).pow(2, cheb=cheb))

    W_fun =( 0.036*(Sw.pow(0.758, cheb=cheb))*Wf.pow(0.0035, cheb=cheb) * 
        (Area * (Lamb.cos(cheb=cheb).pow(2, cheb=cheb)).pow(-1, cheb=cheb)).pow(0.6, cheb=cheb) *
        q.pow(0.006, cheb=cheb) * lamb.pow(0.04, cheb=cheb) * 
        (100 * tc * (Lamb.cos(cheb=cheb)).pow(-1, cheb=cheb)).pow(-0.3, cheb=cheb) *
        (Nz * Wdg).pow(0.49, cheb=cheb) + Sw * Wp
    )    

    return W_fun

X = AffineArray.from_intervals(
    [(0, 1)]*10
)

# Store results per split configuration
results_Branin = {}

# maximum number of splits, 2^dim1 in each direction
dim1 = 3

print("======================================= \n"
    f"Results from Wing-weight-model function \n"
    "=======================================")

for m in range(dim1):

    splits_per_dim = [2**m]*10

    subintervals = X.split_interval(splits_per_dim)

    # Reset accumulators for each (m, n) configuration          
    lo_Ch, hi_Ch = np.inf, -np.inf
    lo_MR, hi_MR = np.inf, -np.inf

    for Xi in subintervals:

        # --- Min-range fit ---

        funVal_MR = WingWeight_model(Xi,cheb=False)

        frange_MR = funVal_MR.interval

        lo_MR = min(lo_MR, frange_MR[0])
        hi_MR = max(hi_MR, frange_MR[1])

        # --- Chebyshev fit ---
        
        funVal_ch = WingWeight_model(Xi, cheb=True)

        frange_ch = funVal_ch.interval

        lo_Ch = min(lo_Ch, frange_ch[0])
        hi_Ch = max(hi_Ch, frange_ch[1])

    # Store result for this (m, n) configuration
    results_Branin[(2**m)*10] = {
        'f_rangeCh' : [lo_Ch, hi_Ch],
        'f_rangeMR' : [lo_MR, hi_MR],
        'f_intersect': [max(lo_Ch, lo_MR), min(hi_Ch, hi_MR)]
    }

    print(f"splits={splits_per_dim}  |  "
        f"Chebyshev: [{lo_Ch:.6f}, {hi_Ch:.6f}]  |  "
        f"Min-range: [{lo_MR:.6f}, {hi_MR:.6f}]  |  "
        f"f_intersect: [{max(lo_Ch, lo_MR):.4f}, {min(hi_Ch, hi_MR):.4f}]")


