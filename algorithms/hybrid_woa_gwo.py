import random

def initialize_population(search_space, num_individuals):
    """ایجاد مقادیر اولیه جمعیت."""
    return [
        {key: random.uniform(*val) if isinstance(val, tuple) else random.choice(val)
         for key, val in search_space.items()}
        for _ in range(num_individuals)
    ]

def evaluate_population(population, fitness_function):
    """محاسبه Fitness برای هر فرد در جمعیت."""
    return [fitness_function(individual) for individual in population]

def select_alpha_beta_delta(population, fitness_scores):
    """انتخاب بهترین سه فرد."""
    sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)
    alpha, beta, delta = population[sorted_indices[0]], population[sorted_indices[1]], population[sorted_indices[2]]
    return alpha, beta, delta

def hybrid_woa_gwo(search_space, fitness_function, num_individuals=20, iterations=50):
    """
    ترکیب الگوریتم WOA و GWO.
    Args:
        search_space (dict): فضای جستجو.
        fitness_function (func): تابع ارزیابی.
        num_individuals (int): تعداد جمعیت.
        iterations (int): تعداد تکرارها.
    Returns:
        dict: بهترین جواب.
    """
    population = initialize_population(search_space, num_individuals)
    fitness_scores = evaluate_population(population, fitness_function)
    alpha, beta, delta = select_alpha_beta_delta(population, fitness_scores)

    for t in range(iterations):
        a = 2 - 2 * (t / iterations)  # پارامتر WOA برای کاهش خطی
        for i in range(num_individuals):
            r1, r2 = random.random(), random.random()
            A, C = 2 * a * r1 - a, 2 * r2

            # به روز رسانی موقعیت هر فرد
            updated_position = {}
            for key in search_space:
                D_alpha = abs(C * float(alpha[key]) - float(population[i][key]))
                D_beta = abs(C * beta[key] - population[i][key])
                D_delta = abs(C * delta[key] - population[i][key])
                
                # میانگین گرفتن برای به روز رسانی موقعیت
                updated_position[key] = (alpha[key] - A * D_alpha + beta[key] - A * D_beta + delta[key] - A * D_delta) / 3
            
            population[i] = updated_position

        fitness_scores = evaluate_population(population, fitness_function)
        alpha, beta, delta = select_alpha_beta_delta(population, fitness_scores)

    return alpha  # بازگشت بهترین فرد
