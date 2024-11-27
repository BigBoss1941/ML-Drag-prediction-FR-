import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd #MAKE SURE TO USE PANDA WITH OLDER NUMPY ALONG COMPATIBLE PYTHON VERSION, LATEST ISN'T COMPATIBLE.
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# Exemple de données pour 9 roues
data={
    'P1': [21.1, 22.034, 42.572, 33.763, 43.617, 27.974, 23, 5, 128.342],
    'P2': [22.45, 44.03, 43.415, 40.543, 41.952, 48.79, 83, 6.913, 163.268],
    'P3': [45, 11.31, 0, 2.11, 0, 41.229, 0, 0, 0],
    'P4': [52.83, 74.578, 24.753, 24.753, 72.5, 28.579, 33.392, 1.413, 186.742],
    'P6': [90, 88, 90, 95, 90, 90, 86, 91, 91],
    'Drag': [0.55, 0.50, 0.85, 0.40, 0.65, 0.60, 0.89, 0.60, 0.50]
}

#Cx coefficient de trainée
# Conversion des données en DataFrame
df=pd.DataFrame(data)
#TODO P1 42.14 , P2 40.67 , P3 7.792 , P4 24.753 , P6 89 , Drag 0.55
#init
degree = 0
betas = []
intercept=0
most_impactful_parameter=''
most_impactful_coefficient=0
poly_features_n = None
model_n = None
def predict_drag_n(model_n, poly_features_n, P1, P2, P3, P4, P6):
    # Création d'une matrice de caractéristiques d'entrée pour la prédiction
    X_pred = np.array([[P1, P2, P3, P4, P6]])
    X_pred_poly = poly_features_n.transform(X_pred)

    # Predict drag
    return model_n.predict(X_pred_poly)[0]

def option_one():
    # Déclaration des variables globales
    global df, degree, betas, intercept, most_impactful_coefficient, most_impactful_parameter
    #P: Parameter. Refer to docs to match which is which.
    P1_test = float(input("Entrez la valeure de P1 pour la roue:"))
    P2_test = float(input("Entrez la valeure de P2 pour la roue:"))
    P3_test = float(input("Entrez la valeure de P3 pour la roue:"))
    P4_test = float(input("Entrez la valeure de P4 pour la roue:"))
    P6_test = float(input("Entrez la valeure de P6 pour la roue:"))
    while (degree != 1 and degree != 2 and degree != 4):
        degree = int(input(
            "Entrez le degré de la regression (1 pour linéaire, 2 pour polynomial de deuxiéme degré, 4 pour polynomial de quatriéme degré):"))  # Choisir le degré de régression polynomiale
    call_polynomial_eq(degree)
    if degree == 1:
        predicted_drag1 = predict_drag_first_degree([P1_test, P2_test, P3_test, P4_test, P6_test], betas, intercept)
        print(f"Prédiction de la trainée: {predicted_drag1:.4f}")
        target = float(input("Entrez la valeure ciblé de la trainée:"))
        dif = -(target - predicted_drag1) / most_impactful_coefficient
        P = [P1_test, P2_test, P3_test, P4_test, P6_test]
        most_impactful_param_pos = int(most_impactful_parameter[1:])
        targetted_param = dif + P[most_impactful_param_pos - 1]
        print('Pour atteindre la trainée souhaitée, il faut modifier le parametre ayant le plus grand impact (' + str(most_impactful_parameter) + ') par cette valeure: ' + str(targetted_param))
        degree = 0

    elif degree == 2 or degree == 4:
        global model_n, poly_features_n #outside scope
        drag_prediction = predict_drag_n(model_n, poly_features_n, P1_test, P2_test, P3_test, P4_test, P6_test)
        print(f"Prédiction de la trainée: {drag_prediction:.4f}")
        degree = 0

def evaluate_expression(expression, variables):#TODO introduce fail check
    # Substitution des variables
    for var, value in variables.items():
        expression = expression.replace(var, str(value))

    try:
        result = eval(expression)
        return result
    except Exception as e:
        print(f"Error evaluating expression: {e}")
        return None

def predict_drag(P, betas, intercept):
    #Éxtraction des parametres
    P1=P[0]
    P2=P[1]
    P3=P[2]
    P4=P[3]
    P6=P[4]
    variables = {'P1':P1, 'P2':P2, 'P3':P3, 'P4':P4, 'P6':P6} #TODO Refactor
    linear_terms = betas[1:6]
    quadratic_terms = betas[6:11]
    interaction_terms = betas[11:]

    linear_sum = linear_terms[0]*P1 + linear_terms[1]*P2 + linear_terms[2]*P3 + linear_terms[3]*P4 + linear_terms[4]*P6

    quadratic_sum = quadratic_terms[0]*(P1**2) + quadratic_terms[1]*(P2**2) + quadratic_terms[2]*(P3**2) + quadratic_terms[3]*(P4**2) + quadratic_terms[4]*(P6**2)

    # éxtraite de l'équation de régression, pour améliorer la pérformance
    interactions = [
        ('P1*P2', 0), ('P1*P3', 1), ('P1*P4', 2), ('P1*P6', 3),
        ('P2*P3', 4), ('P2*P4', 5), ('P2*P6', 6),
        ('P3*P4', 7), ('P3*P6', 8),
        ('P4*P6', 9)
    ]
    interactions_sum = interaction_terms[0]*P1*P2 + interaction_terms[1]*P1*P3 + interaction_terms[2]*P1*P4 + interaction_terms[3]*P1*P6 + interaction_terms[4]*P2*P3 + interaction_terms[5]*P2*P4 + interaction_terms[6]*P2*P6 + interaction_terms[7]*P3*P4 + interaction_terms[8]*P3*P6 + interaction_terms[9]*P4*P6

    predicted_drag = linear_sum + quadratic_sum + interactions_sum
    print(betas)#TODO remove DEBUG
    return predicted_drag

def predict_drag_first_degree(P, betas, intercept):
    predicted_drag = intercept + betas[0]*P[0] + betas[1]*P[1] + betas[2]*P[2] + betas[3]*P[3] + betas[4]*P[4]

    return predicted_drag


def option_two():
    # déclaration des variables globales
    global df, degreepoly_featurespoly_features
    P1_7 = float(input("Entrez la valeure de P1 pour la Nouvelle roue:"))
    P2_7 = float(input("Entrez la valeure de P2 pour la Nouvelle roue:"))
    P3_7 = float(input("Entrez la valeure de P3 pour la Nouvelle roue:"))
    P4_7 = float(input("Entrez la valeure de P4 pour la Nouvelle roue:"))
    P6_7 = float(input("Entrez la valeure de P6 pour la Nouvelle roue:"))
    Drag_7 = float(input("Entrez la valeure de la trainée (Drag) pour la Nouvelle roue:"))
    df = df._append({'P1': P1_7, 'P2': P2_7, 'P3': P3_7, 'P4': P4_7, 'P6': P6_7, 'Drag': Drag_7}, ignore_index=True)
    while(degree != 1 and degree != 2 and degree != 4):
        degree = int(input("Entrez le degré de la regression (1 pour linéaire, 2 pour polynomial de deuxiéme degré, 4 pour polynomial de quatriéme degré):"))  # Choisir le degré de régression polynomiale
    call_polynomial_eq(degree)

def call_polynomial_eq(degree):
    global df, betas, intercept, most_impactful_parameter, most_impactful_coefficient
    if degree == 1:
        X = df[['P1', 'P2', 'P3', 'P4', 'P6']]
        y = df['Drag']

        # Création et ajustement du modéle de régression linéaire
        model = LinearRegression()
        model.fit(X, y)

        # Recupération des coefficient set de l’intercept
        betas = model.coef_
        intercept = model.intercept_

        # Prédiction
        y_pred = model.predict(X)

        # Évaluation du modéle
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        # Affichage des résultats
        print(f" Coefficients β i: {betas}")
        print(f" Intercept: {intercept}")
        print(f" Mean Squared Error: {mse}")
        print(f" R−squared: {r2} ")
        print(
            f" Function Objective: Drag  =  {intercept}+" + "+".join([f"({b})*P{i + 1}" for i, b in enumerate(betas)]))

        # Déterminer le paramétre avec le plus grand impact
        impact = {f'P{i + 1}': abs(b) for i, b in enumerate(betas)}
        most_impactful_parameter = max(impact, key=impact.get)
        most_impactful_coefficient = impact[most_impactful_parameter]

        # Trier les paramétres en fonction de leur impact
        sorted_params = sorted(impact.items(), key=lambda x: x[1], reverse=True)

        # Afficher les paramétres et leur impact
        print("Coefficients des paramétres tris par impact:")

        for param, coeff in sorted_params:
            if coeff >= most_impactful_coefficient / 2:
                impact_level = "high impact"
            else:
                impact_level = "low impact"
            print(f"{param}:   {coeff}   ({impact_level})")
    elif degree == 2:
        # Séparation des variables indpendantes (P1 P7) et de la variable dépendante ( Drag )
        global model_n, poly_features_n
        X_unpoly = df[['P1', 'P2', 'P3', 'P4', 'P6']]
        y = df['Drag']

        # Appel d'une regression polynomiale de degré défini par une variable
        poly_features_n = PolynomialFeatures(degree=degree)
        X = poly_features_n.fit_transform(X_unpoly)
        # Création et ajustement du modéle de régression linéaire
        model_n = LinearRegression()
        model_n.fit(X, y)

        # Recupération des coefficient set de l’intercept
        betas = model_n.coef_
        intercept = model_n.intercept_

        # Prédictions
        y_pred = model_n.predict(X)

        # Évaluation du modéle
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        # Affichage des résultats
        print(f" Coefficients βi: {betas}")
        print(f" Intercept: {intercept}")
        print(f" Mean Squared Error: {mse}")
        print(f" R−squared: {r2} ")

        def print_polynomial_equation(betas, intercept):

            var_names = ['P1', 'P2', 'P3', 'P4', 'P6']


            equation = f"Drag = {betas[0]}"

            for i in range(5):
                if betas[i + 1] != 0:
                    equation += f" + ({betas[i + 1]} * {var_names[i]}^1)"


            for i in range(5):
                if betas[i + 6] != 0:
                    equation += f" + ({betas[i + 6]} * {var_names[i]}^2)"

            # Ajout termes d'interaction. Le pool est issue d'un développement de l'équation de la régression polynomiale, pour un temps d'éxecution plus rapide
            interactions = [
                ('P1*P2', 11), ('P1*P3', 12), ('P1*P4', 13), ('P1*P6', 14),
                ('P2*P3', 15), ('P2*P4', 16), ('P2*P6', 17),
                ('P3*P4', 18), ('P3*P6', 19),
                ('P4*P6', 20)
            ]
            for term, index in interactions:
                if betas[index] != 0:
                    equation += f" + ({betas[index]} * {term})"

            return equation

        # actual operation
        print(print_polynomial_equation(betas, intercept))
        # Déterminer le paramétre avec le plus grand impact
        impact = {f'P{i + 1}': abs(b) for i, b in enumerate(betas)}
        most_impactful_parameter = max(impact, key=impact.get)
        most_impactful_coefficient = impact[most_impactful_parameter]

        # Trier les paramétres en fonction de leur impact
        sorted_params = sorted(impact.items(), key=lambda x: x[1], reverse=True)

        # Afficher les paramtres et leur impact
        print("Paramtres tris par impact:")

        for param, coeff in sorted_params:
            if coeff >= most_impactful_coefficient / 2:
                impact_level = "high impact"
            else:
                impact_level = "low impact"
            print(f"{param.lower()}:   {coeff}   ({impact_level})")

    elif degree == 4:
        # Séparation des variables indpendantes (P1 P7) et de la variable dépendante ( Drag )
        X_unpoly = df[['P1', 'P2', 'P3', 'P4', 'P6']]
        y = df['Drag']

        # Appél d'une regression polynomiale de degré défini par une variable
        poly_features_n = PolynomialFeatures(degree=degree)
        X = poly_features_n.fit_transform(X_unpoly)
        # Création et ajustement du modéle de régression linéaire
        model_n = LinearRegression()
        model_n.fit(X, y)

        # Recupération des coefficient set de l’intercept
        betas = model_n.coef_
        intercept = model_n.intercept_

        # Prédictions
        y_pred = model_n.predict(X)

        # évaluation du modéle
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        # Affichage des résultats
        print(f" Coefficients βi: {betas}")
        print(f" Intercept: {intercept}")
        print(f" Mean Squared Error: {mse}")
        print(f" R−squared: {r2} ")

        def print_fourth_degree_polynomial_equation(betas):
            # Define the names of the variables
            var_names = ['P1', 'P2', 'P3', 'P4', 'P6']

            # Initialize the equation string
            equation = f"Drag = {betas[0]}"  # Start with the intercept

            # Ajout termes linéaires
            for i in range(5):
                if betas[i + 1] != 0:
                    equation += f" + ({betas[i + 1]} * {var_names[i]}^1)"

            # Ajouter termes quadratiques
            for i in range(5):
                if betas[i + 6] != 0:
                    equation += f" + ({betas[i + 6]} * {var_names[i]}^2)"

            # Ajout termes cubiques
            for i in range(5):
                if betas[i + 11] != 0:
                    equation += f" + ({betas[i + 11]} * {var_names[i]}^3)"

            # Ajout termes quatrième degré
            for i in range(5):
                if betas[i + 16] != 0:
                    equation += f" + ({betas[i + 16]} * {var_names[i]}^4)"

            interactions = [
                ('P1*P2', 21), ('P1*P3', 22), ('P1*P4', 23), ('P1*P6', 24),
                ('P2*P3', 25), ('P2*P4', 26), ('P2*P6', 27),
                ('P3*P4', 28), ('P3*P6', 29),
                ('P4*P6', 30)
            ]
            for term, index in interactions:
                if betas[index] != 0:
                    equation += f" + ({betas[index]} * {term})"

            quad_interactions = [
                ('P1^2*P2', 31), ('P1^2*P3', 32), ('P1^2*P4', 33), ('P1^2*P6', 34),
                ('P1*P2^2', 35), ('P1*P3^2', 36), ('P1*P4^2', 37), ('P1*P6^2', 38),
                ('P2^2*P3', 39), ('P2^2*P4', 40), ('P2^2*P6', 41),
                ('P2*P3^2', 42), ('P2*P4^2', 43), ('P2*P6^2', 44),
                ('P3^2*P4', 45), ('P3^2*P6', 46),
                ('P3*P4^2', 47), ('P3*P6^2', 48),
                ('P4^2*P6', 49)
            ]
            for term, index in quad_interactions:
                if betas[index] != 0:
                    equation += f" + ({betas[index]} * {term})"

            # Pool termes cubiques
            cub_interactions = [
                ('P1^3*P2', 50), ('P1^3*P3', 51), ('P1^3*P4', 52), ('P1^3*P6', 53),
                ('P1^2*P2^2', 54), ('P1^2*P3^2', 55), ('P1^2*P4^2', 56), ('P1^2*P6^2', 57),
                ('P1*P2^3', 58), ('P1*P3^3', 59), ('P1*P4^3', 60), ('P1*P6^3', 61),
                ('P2^3*P3', 62), ('P2^3*P4', 63), ('P2^3*P6', 64),
                ('P2^2*P3^2', 65), ('P2^2*P4^2', 66), ('P2^2*P6^2', 67),
                ('P2*P3^3', 68), ('P2*P4^3', 69), ('P2*P6^3', 70),
                ('P3^3*P4', 71), ('P3^3*P6', 72),
                ('P3^2*P4^2', 73), ('P3^2*P6^2', 74),
                ('P3*P4^3', 75), ('P3*P6^3', 76),
                ('P4^3*P6', 77)
            ]
            for term, index in cub_interactions:
                if betas[index] != 0:
                    equation += f" + ({betas[index]} * {term})"

            # Ajout termes d'interaction de quatrième degré (aussi éxtraite)
            fourth_interactions = [
                ('P1^2*P2*P3', 78), ('P1^2*P2*P4', 79), ('P1^2*P2*P6', 80),
                ('P1^2*P3*P4', 81), ('P1^2*P3*P6', 82),
                ('P1^2*P4*P6', 83),
                ('P1*P2^2*P3', 84), ('P1*P2^2*P4', 85), ('P1*P2^2*P6', 86),
                ('P1*P2*P3^2', 87), ('P1*P2*P4^2', 88), ('P1*P2*P6^2', 89),
                ('P1*P3^2*P4', 90), ('P1*P3^2*P6', 91),
                ('P1*P3*P4^2', 92), ('P1*P3*P6^2', 93),
                ('P1*P4^2*P6', 94),
                ('P2^2*P3*P4', 95), ('P2^2*P3*P6', 96),
                ('P2^2*P4*P6', 97),
                ('P2*P3^2*P4', 98), ('P2*P3^2*P6', 99),
                ('P2*P3*P4^2', 100), ('P2*P3*P6^2', 101),
                ('P2*P4^2*P6', 102),
                ('P3^2*P4*P6', 103),
                ('P3*P4^2*P6', 104)
            ]
            for term, index in fourth_interactions:
                if betas[index] != 0:
                    equation += f" + ({betas[index]} * {term})"

            return equation

        # actual operation
        print(print_fourth_degree_polynomial_equation(betas))
        # Déterminer le paramétre avec le plus grand impact
        impact = {f'P{i + 1}': abs(b) for i, b in enumerate(betas)}
        most_impactful_parameter = max(impact, key=impact.get)
        most_impactful_coefficient = impact[most_impactful_parameter]

        # Trier les paramétres en fonction de leur impact
        sorted_params = sorted(impact.items(), key=lambda x: x[1], reverse=True)

        # Afficher les paramétres et leur impact
        print("Paramtres tris par impact:")

        for param, coeff in sorted_params:
            if coeff >= most_impactful_coefficient / 2:
                impact_level = "high impact"
            else:
                impact_level = "low impact"
            print(f"{param.lower()}:   {coeff}   ({impact_level})")

#Menu
def main_menu():
    #boucle menu
    global degree
    while True:
        print("\nMenu programme regression")
        print("1. Programme de prediction de trainée")
        print("2. Ajouter des données dans la base de données")
        print("3. Afficher l'équation de la régression")
        print("0. Sortir")

        choice = input("Choisir une option (0-3): ")

        if choice == "1":
            option_one()
        elif choice == "2":
            option_two()
        elif choice == "3":
            while (degree != 1 and degree != 2 and degree != 4):
                degree = int(input(
                    "Entrez le degré de la regression (1 pour linéaire, 2 pour polynomial de deuxiéme degré, 4 pour polynomial de quatriéme degré):"))  # Choisir le degré de régression polynomiale
            call_polynomial_eq(degree)
            degree = 0
        elif choice == "0":
            print("Fin de programme")
            break
        else:
            print("Choix non valide. Veuillez réessayer.")

#Appel du menu
main_menu()


