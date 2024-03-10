import numpy as np
from PIL import Image

def get_reconstructed_image(image, rank):
    # Set the desired number of singular values to keep (k)
    k = rank
    image = np.array(image)
    # Perform SVD
    reconstructed_matrix = []
    parameters_og = image.shape[0] * image.shape[1] * image.shape[2]
    parameters_lora = 0
    for i in range(image.shape[2]):
        print("Rank of the matrix: ", np.linalg.matrix_rank(image[:, :, i]))
        U, S, VT = np.linalg.svd(image[:, :, i], full_matrices=False)

        # Keep only the largest k singular values and corresponding columns of U and VT
        U_k = U[:, :k]
        S_k = np.diag(S[:k])
        VT_k = VT[:k, :]
        # Reconstruct the original matrix using the truncated matrices
        parameters_lora += U_k.shape[0] * U_k.shape[1] + S_k.shape[0] * S_k.shape[1] + VT_k.shape[0] * VT_k.shape[1]
        reconstructed_matrix.append(np.dot(U_k, np.dot(S_k, VT_k)))
    reconstructed_image = np.dstack(reconstructed_matrix)
    reconstructed_image = np.clip(reconstructed_image, 0, 255)
    reconstructed_image = Image.fromarray(reconstructed_image.astype(np.uint8))
    print("Parameters Original: ", parameters_og)
    print("Parameters LoRA: ", parameters_lora)
    return reconstructed_image


if __name__=="__main__":
    # Load the image
    PATH = "../../../datasets/lora_reconstr/"
    image = Image.open(f"{PATH}flower.jpeg")
    
    
    # Reconstruct the image using the first 10 singular values
    rank = 75
    reconstructed_image = get_reconstructed_image(image, rank)

    # Save the reconstructed image
    reconstructed_image.save(f"reconstructed_flower_{rank}.jpg")
    # reconstructed_image.show()