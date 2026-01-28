from hardware.manta_driver import MantaDriver
from analysis.fitter import InterferenceFitter
import matplotlib.pyplot as plt
import numpy as np

def main():
    # 1. Connect and Snap
    driver = MantaDriver() # Add ID if needed
    driver.connect()
    
    driver.set_exposure(0.005) # 5ms
    driver.set_gain(0)
    
    print("Acquiring image...")
    img = driver.acquire_frame()
    driver.close()
    
    # 2. Analyze
    fitter = InterferenceFitter()
    
    # Collapse to 1D
    lineout = fitter.get_lineout(img)
    
    # Fit
    print("Fitting data...")
    res = fitter.fit(lineout)
    
    if res:
        print(f"FIT SUCCESS!")
        print(f"Visibility: {res['visibility']:.4f}")
        print(f"Beam Sigma: {res['sigma']*1e6:.2f} microns")
        
        # 3. Plot Result
        plt.figure(figsize=(10, 6))
        plt.plot(lineout, label='Raw Data (Summed)', color='black', alpha=0.6)
        plt.plot(res['fitted_curve'], label='Fit', color='red', linestyle='--')
        plt.legend()
        plt.title(f"Interference Fit (Vis={res['visibility']:.2f})")
        plt.show()
    else:
        print("Fit Failed! (Maybe image is too dark or empty?)")
        plt.plot(lineout)
        plt.show()

if __name__ == "__main__":
    main()