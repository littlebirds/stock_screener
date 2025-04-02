{
  "cells": [
    {
      "cell_type": "code",
      "source": [
        "!git clone https://github.com/littlebirds/stock_screener.git"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "V2wajILH1vxU",
        "outputId": "7a57334f-d003-4230-a70c-d0884718b17d"
      },
      "id": "V2wajILH1vxU",
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Cloning into 'stock_screener'...\n",
            "remote: Enumerating objects: 63, done.\u001b[K\n",
            "remote: Counting objects: 100% (63/63), done.\u001b[K\n",
            "remote: Compressing objects: 100% (45/45), done.\u001b[K\n",
            "remote: Total 63 (delta 23), reused 47 (delta 13), pack-reused 0 (from 0)\u001b[K\n",
            "Receiving objects: 100% (63/63), 149.48 KiB | 3.74 MiB/s, done.\n",
            "Resolving deltas: 100% (23/23), done.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "%cd stock_screener/"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "j0cKaNlA2cMz",
        "outputId": "0706b83d-6d67-4e0c-fe8a-559db5757d5f"
      },
      "id": "j0cKaNlA2cMz",
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[Errno 2] No such file or directory: 'stock_screener/'\n",
            "/content\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 44,
      "id": "08c7cabf-e6f6-4b49-9192-96b749e39b82",
      "metadata": {
        "id": "08c7cabf-e6f6-4b49-9192-96b749e39b82"
      },
      "outputs": [],
      "source": [
        "import datetime\n",
        "\n",
        "from datasets import SpyDailyDataset\n",
        "from models import Conv1DNet"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 49,
      "id": "wOdupmDKjfmu",
      "metadata": {
        "id": "wOdupmDKjfmu"
      },
      "outputs": [],
      "source": [
        "ds = SpyDailyDataset()\n",
        "s = ds[30]"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pwd"
      ],
      "metadata": {
        "id": "KaKkFJgUzELd",
        "outputId": "5ad7f8bd-a9f5-4199-d8c6-171baf586ef9",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 35
        }
      },
      "id": "KaKkFJgUzELd",
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "'/content'"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            }
          },
          "metadata": {},
          "execution_count": 2
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "27d5cb37-8dc0-4127-99eb-854476879257",
      "metadata": {
        "id": "27d5cb37-8dc0-4127-99eb-854476879257"
      },
      "outputs": [],
      "source": [
        "def training_loop(n_epochs, optimizer, model, loss_fn, train_loader):\n",
        "    for epoch in range(1, n_epochs+1):\n",
        "        loss_train = 0.0\n",
        "        for batch, lables in train_loader:\n",
        "            outputs  = model(batch)\n",
        "            loss = loss_fn(outputs, labels)\n",
        "            optimizer.zero_grad()\n",
        "            loss.backward()\n",
        "            optimizer.step()\n",
        "            loss_train += loss.item()\n",
        "\n",
        "        if epoch ==1 or epoch % 10 == 0:\n",
        "            print('{} Epoch {}, training loss {}'.format(datetime.datatime.now(), epoch, loss_train/len(train_loader)))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "b656f60d-8744-44e8-a748-ade47bf0951b",
      "metadata": {
        "id": "b656f60d-8744-44e8-a748-ade47bf0951b"
      },
      "outputs": [],
      "source": []
    },
    {
      "cell_type": "code",
      "execution_count": 26,
      "id": "a8aeadd6-6821-414a-a088-688ef558338e",
      "metadata": {
        "id": "a8aeadd6-6821-414a-a088-688ef558338e",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "15f72fe8-b83f-49c3-9da5-8106da5b3038"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "tensor([[0.4624, 0.9952, 0.7824],\n",
            "        [0.3294, 0.9905, 0.4125],\n",
            "        [0.7482, 0.9372, 0.4367],\n",
            "        [0.1497, 0.3710, 0.1434],\n",
            "        [0.4383, 0.9364, 0.7651],\n",
            "        [0.8642, 0.1288, 0.3233],\n",
            "        [0.4317, 0.3357, 0.4963],\n",
            "        [0.5220, 0.7320, 0.6103],\n",
            "        [0.8372, 0.3165, 0.1871],\n",
            "        [0.8719, 0.6985, 0.1758]])\n"
          ]
        }
      ],
      "source": [
        "import torch\n",
        "n = torch.rand(10,3)\n",
        "print(n)"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(n)\n",
        "m = n[-1, 1:]\n",
        "print(m.shape)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "b2Mfx8sstT9r",
        "outputId": "12929736-5be4-45a0-a7ff-7fe1b1b87147"
      },
      "id": "b2Mfx8sstT9r",
      "execution_count": 43,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "tensor([[0.4624, 0.9952, 0.7824],\n",
            "        [0.3294, 0.9905, 0.4125],\n",
            "        [0.7482, 0.9372, 0.4367],\n",
            "        [0.1497, 0.3710, 0.1434],\n",
            "        [0.4383, 0.9364, 0.7651],\n",
            "        [0.8642, 0.1288, 0.3233],\n",
            "        [0.4317, 0.3357, 0.4963],\n",
            "        [0.5220, 0.7320, 0.6103],\n",
            "        [0.8372, 0.3165, 0.1871],\n",
            "        [0.8719, 0.6985, 0.1758]])\n",
            "torch.Size([2])\n"
          ]
        }
      ]
    }
  ],
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3 (ipykernel)",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.12.3"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}