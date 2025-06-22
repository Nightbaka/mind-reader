import numpy as np
from scipy.linalg import sqrtm

def fit_n_dimensional_gaussian(latents):
    """
    Funkcja dopasowująca rozkład Gaussa o parametrach średniej mu i macierzy kowariancji sigma do danych w przestrzeni ukrytej

    :latents: dane w przestrzeni ukrytej, dla których dopasowujemy rozkład Gaussa

    :return: mu, sigma - parametry rozkładu Gaussa - wektor średniej i macierz kowariancji
    """
    # POCZĄTEK ROZWIĄZANIA
    mu = np.mean(latents, axis=0)
    sigma = np.cov(latents, rowvar=False)
    # KONIEC ROZWIĄZANIA
    return mu, sigma


def wasserstein_distance(mu1, sigma1, mu2, sigma2):
    """
    Funkcja wyznaczająca odległość Wassersteina pomiędzy dwoma rozkładami Gaussa o parametrach mu i sigma

    :mu1: średnia pierwszego rozkładu
    :sigma1: macierz kowariancji pierwszego rozkładu
    :mu2: średnia drugiego rozkładu
    :sigma2: macierz kowariancji drugiego rozkładu
    """
    # POCZĄTEK ROZWIĄZANIA
    diff = mu1 - mu2
    norm_squared = np.dot(diff, diff)

    # Macierzowy pierwiastek z iloczynu
    sqrt_product = sqrtm(sigma1 @ sigma2)

    # błąd numeryczny może sprawić, że wynik jest zespolony, dlatego interesuje nas tylko rzeczywista część
    if np.iscomplexobj(sqrt_product):
        sqrt_product = sqrt_product.real

    trace_component = np.trace(sigma1 + sigma2 - 2 * sqrt_product)
    distance = norm_squared + trace_component
    # KONIEC ROZWIĄZANIA

    return distance
