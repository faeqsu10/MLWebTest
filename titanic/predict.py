import pickle

def getPrediction(pclass, sex, age, sibsp, parch, fare, C, Q, S):
    model = pickle.load(open('titanic_lr_model.pkl', 'rb'))
    scaled = pickle.load(open('titanic_minmax_scaler.pkl', 'rb'))
    transform = scaled.transform([[pclass, sex, age, sibsp, parch, fare, C, Q, S]])
    prediction = model.predict(transform)

    return '사망' if prediction == 0 else '생존' if prediction == 1 else 'error'