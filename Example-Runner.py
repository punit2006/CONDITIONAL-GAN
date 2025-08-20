from cgan.train_basic import train_basic
from cgan.train_advanced import train_advanced

if __name__ == "__main__":
    print("Running Basic CGAN...")
    train_basic(epochs=500)

    print("\nRunning Advanced CGAN...")
    train_advanced(epochs=2000)
