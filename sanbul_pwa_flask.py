from flask import Flask, render_template, redirect, url_for, request, session
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
from flask_bootstrap import Bootstrap5
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras

# Flask 앱 설정
app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'

bootstrap5 = Bootstrap5(app)

# 입력 폼 정의
class LabForm(FlaskForm):
    longitude = StringField('longitude (1~7)', validators=[DataRequired()])
    latitude = StringField('latitude (1~7)', validators=[DataRequired()])
    month = StringField('month (01-Jan ~ Dec-12)', validators=[DataRequired()])
    day = StringField('day (00-sun ~ 06-sat, 07-hol)', validators=[DataRequired()])
    avg_temp = StringField('avg_temp', validators=[DataRequired()])
    max_temp = StringField('max_temp', validators=[DataRequired()])
    max_wind_speed = StringField('max_wind_speed', validators=[DataRequired()])
    avg_wind = StringField('avg_wind', validators=[DataRequired()])
    submit = SubmitField('Submit')

# index 페이지
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

# prediction 페이지 (입력 폼만)
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    form = LabForm()
    if form.validate_on_submit():
        # 입력 데이터를 세션에 저장
        session['input_data'] = {
            'longitude': float(form.longitude.data),
            'latitude': float(form.latitude.data),
            'month': form.month.data,
            'day': form.day.data,
            'avg_temp': float(form.avg_temp.data),
            'max_temp': float(form.max_temp.data),
            'max_wind_speed': float(form.max_wind_speed.data),
            'avg_wind': float(form.avg_wind.data)
        }
        return redirect(url_for('result'))
    return render_template('prediction.html', form=form)

# result 페이지 (예측 수행 및 결과 출력)
@app.route('/result')
def result():
    try:
        input_data = session.get('input_data')
        if not input_data:
            return redirect(url_for('prediction'))

        # 입력값을 DataFrame으로 변환
        input_df = pd.DataFrame([input_data])

        # 전처리 파이프라인 및 모델 불러오기
        full_pipeline = joblib.load('full_pipeline.pkl')
        model = keras.models.load_model('best_fire_model.keras')

        # 데이터 전처리 및 예측
        input_prepared = full_pipeline.transform(input_df)
        pred_log = model.predict(input_prepared)[0][0]
        prediction = np.expm1(pred_log)  # 로그 변환 역변환

        return render_template('result.html', prediction=round(prediction, 2))

    except Exception as e:
        return f"Error during predict: {e}"

if __name__ == '__main__':
    app.run(debug=True)

