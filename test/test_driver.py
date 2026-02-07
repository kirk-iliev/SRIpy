import logging
from hardware.manta_driver import MantaDriver
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

def main():
    driver = MantaDriver()
    
    try:
        driver.connect()
        # Increase exposure to 500ms (0.5s) to gather more light
        driver.exposure = 0.5 
        driver.gain = 20  # Crank the gain just for this test
        
        logger.info("Acquiring image...")
        img = driver.acquire_frame()
        
        # Squeeze the extra dimension: (1216, 1936, 1) -> (1216, 1936)
        img = img.squeeze()
        
        logger.info(f"Stats: Min={img.min()}, Max={img.max()}, Mean={img.mean():.2f}")
        
        # Plot with "auto-scaling" based on percentile to ignore hot pixels
        # This forces the colormap to stretch effectively
        vmin, vmax = np.percentile(img, (1, 99))
        
        plt.figure(figsize=(10, 8))
        plt.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
        plt.colorbar(label='Intensity')
        plt.title(f"Manta Cam Test (Exp: 500ms, Mean: {img.mean():.2f})")
        plt.show()
        
    finally:
        driver.close()

if __name__ == "__main__":
    main()