import timm
import torch


def main():
    print("Hello from self-supervised-timm!")
    print(timm.__version__)
    print("List of models:")
    print(timm.list_models())
    print(f"CUDA available: {torch.cuda.is_available()}")


if __name__ == "__main__":
    main()
