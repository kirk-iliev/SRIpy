import logging
from hardware.manta_driver import MantaDriver
from analysis.fitter import InterferenceFitter
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

def main():
    # 1. Connect and Snap
    driver = MantaDriver() # Add ID if needed
    driver.connect()
    
    driver.exposure = 0.005  # 5ms
    driver.gain = 0
    
    logger.info("Acquiring image...")
    img = driver.acquire_frame()
    driver.close()
    
    # 2. Analyze
    fitter = InterferenceFitter()
    
    # Collapse to 1D
    lineout = fitter.get_lineout(img)
    
    # Fit
    logger.info("Fitting data...")
    res = fitter.fit(lineout)
    
    if res:
        logger.info("FIT SUCCESS!")
        logger.info(f"Visibility: {res['visibility']:.4f}")
        logger.info(f"Beam Sigma: {res['sigma']*1e6:.2f} microns")
        
        # 3. Plot Result
        plt.figure(figsize=(10, 6))
        plt.plot(lineout, label='Raw Data (Summed)', color='black', alpha=0.6)
        plt.plot(res['fitted_curve'], label='Fit', color='red', linestyle='--')
        plt.legend()
        plt.title(f"Interference Fit (Vis={res['visibility']:.2f})")
        plt.show()
    else:
        logger.warning("Fit Failed! (Maybe image is too dark or empty?)")
        plt.plot(lineout)
        plt.show()

if __name__ == "__main__":
    main()