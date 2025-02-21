import os
import pickle
import streamlit as st
import numpy as np
import logging
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

#environment variables
load_dotenv()


if not os.getenv("GOOGLE_API_KEY"):
    st.error("Gemini API token not found. Please set Gemini_API_TOKEN in .env file.")
    llm = None
    model = None
else:
    try:
        model= ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    except Exception as e:
        st.error(f"Failed to initialize language model: {str(e)}")
        llm = None
        model = None

# Set page  
st.set_page_config(page_title="Health Assistant",
                  layout="wide",
                  page_icon="🧑‍⚕️")

# Determine working director
try:
    working_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    working_dir = os.getcwd()

# Load models and scalers  
def load_file(file_path, file_type="model"):
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error(f"{file_type.capitalize()} file not found: {file_path}")
        raise
    except Exception as e:
        st.error(f"Error loading {file_type} file {file_path}: {str(e)}")
        raise

models = {}
scalers = {}
try:
    models['diabetes'] = load_file(f'{working_dir}/saved_models/diabetes_model.sav')
    scalers['diabetes'] = load_file(f'{working_dir}/saved_models/diabetes_scaler.sav', "scaler")
    models['heart_disease'] = load_file(f'{working_dir}/saved_models/heart_disease_model.pkl')
    models['parkinsons'] = load_file(f'{working_dir}/saved_models/parkinsons_model.pkl')
    models['breast_cancer'] = load_file(f'{working_dir}/saved_models/breast_cancer_model.pkl')
    scalers['breast_cancer'] = load_file(f'{working_dir}/saved_models/breast_cancer_scaler.sav', "scaler")
    models['tuberculosis'] = load_file(f'{working_dir}/saved_models/tuberculosis_model.pkl')
    scalers['tuberculosis'] = load_file(f'{working_dir}/saved_models/tuberculosis_scaler.sav', "scaler")
    models['liver'] = load_file(f'{working_dir}/saved_models/liver_model.pkl')
    scalers['liver'] = load_file(f'{working_dir}/saved_models/liver_scaler.sav', "scaler")
    models['stroke'] = load_file(f'{working_dir}/saved_models/stroke_model.pkl')
    scalers['stroke'] = load_file(f'{working_dir}/saved_models/stroke_scaler.sav', "scaler")
except Exception as e:
    st.error("Failed to load all required models. Application may not function correctly.")
    logger.error(f"Model loading error: {str(e)}")

# Enhanced input validation
def safe_float_convert(value, default=0.0, min_val=None, max_val=None, param_name=""):
    if value == '':
        return default
    try:
        val = float(value)
        if min_val is not None and val < min_val:
            
            return min_val
        if max_val is not None and val > max_val:
             
            return max_val
        return val
    except ValueError:
        st.error(f"Invalid input for {param_name}: '{value}' is not a valid number.")
        return default

# Disease report generation
def generate_disease_report(disease_name, prediction_result):
    if model is None:
        return "Report generation unavailable due to language model initialization failure."
    prompt = f"""
    The person has been diagnosed with {disease_name}. 
    Generate a detailed report including:
    1. A brief description of the disease.
    2. 5 precautions to manage or prevent worsening of the condition.
    3. Key areas to work on to stay healthy with this disease.
    """
    try:
        result = model.invoke(prompt)
        return result.content
    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        return f"Error generating report: {str(e)}"

# Sidebar menu
with st.sidebar:
    selected = option_menu('Medical Diagnosis using AI',
                          ['Diabetes Prediction',
                           'Heart Disease Prediction',
                           'Parkinsons Prediction',
                           'Breast Cancer Prediction',
                           'Tuberculosis Prediction',
                           'Liver Disease Prediction',
                           'Stroke Prediction'],
                          menu_icon='hospital-fill',
                          icons=['activity', 'heart', 'person', 'gender-female',
                                 'lungs', 'liver', 'droplet'],
                          default_index=0)

# Diabetes Prediction
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction')
    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies', '1', help="0-20")   
        SkinThickness = st.text_input('Skin Thickness value', '20', help="0-99 mm")  
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value', '0.5', help="0-2.5")  
    with col2:
        Glucose = st.text_input('Glucose Level', '100', help="0-200 mg/dL")  
        Insulin = st.text_input('Insulin Level', '50', help="0-846 mu U/ml")  
        Age = st.text_input('Age of the Person', '30', help="0-120 years")   
    with col3:
        BloodPressure = st.text_input('Blood Pressure value', '70', help="0-122 mmHg")   
        BMI = st.text_input('BMI value', '25', help="10-60")   

    if st.button('Diabetes Test Result'):
        try:
            user_input = [
                safe_float_convert(Pregnancies, min_val=0, max_val=20, param_name="Pregnancies"),
                safe_float_convert(Glucose, min_val=0, max_val=200, param_name="Glucose"),
                safe_float_convert(BloodPressure, min_val=0, max_val=122, param_name="Blood Pressure"),
                safe_float_convert(SkinThickness, min_val=0, max_val=99, param_name="Skin Thickness"),
                safe_float_convert(Insulin, min_val=0, max_val=846, param_name="Insulin"),
                safe_float_convert(BMI, min_val=10, max_val=60, param_name="BMI"),
                safe_float_convert(DiabetesPedigreeFunction, min_val=0, max_val=2.5, param_name="Diabetes Pedigree"),
                safe_float_convert(Age, min_val=0, max_val=120, param_name="Age")
            ]
            input_array = np.array(user_input).reshape(1, -1)
            standardized_input = scalers['diabetes'].transform(input_array)
            prediction = models['diabetes'].predict(standardized_input)
            logger.debug(f"Diabetes input: {user_input}, Prediction: {prediction}")
            if prediction[0] == 1:
                st.success('The person is diabetic')
                report = generate_disease_report("Diabetes", "positive")
                st.subheader("Diagnosis Report")
                st.markdown(report)
            else:
                st.success('The person is not diabetic')
                st.write("No disease detected, hence no report generated.")
        except Exception as e:
            st.error(f"Diabetes prediction failed: {str(e)}")
            logger.error(f"Diabetes prediction error: {str(e)}")

# Heart Disease Prediction
elif selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction')
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.text_input('Age', '50', help="0-120")   
        cp = st.text_input('Chest Pain types', '1', help="0-3")   
        trestbps = st.text_input('Resting Blood Pressure', '120', help="94-200 mmHg")   
    with col2:
        sex = st.text_input('Sex', '1', help="0 (female) or 1 (male)")   
        chol = st.text_input('Serum Cholestoral in mg/dl', '200', help="126-564 mg/dl")   
        restecg = st.text_input('Resting Electrocardiographic results', '0', help="0-2")   
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl', '0', help="0 or 1")   
        thalach = st.text_input('Maximum Heart Rate achieved', '150', help="71-202 bpm")   
        exang = st.text_input('Exercise Induced Angina', '0', help="0 or 1")  

    col4, col5 = st.columns(2)
    with col4:
        oldpeak = st.text_input('ST depression induced by exercise', '1.0', help="0-6.2")   
        slope = st.text_input('Slope of the peak exercise ST segment', '1', help="0-2")   
    with col5:
        ca = st.text_input('Major vessels colored by fluoroscopy', '0', help="0-4")   
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversible defect', '2', help="0-2")  

    if st.button('Heart Disease Test Result'):
        try:
            user_input = [
                safe_float_convert(age, min_val=0, max_val=120, param_name="Age"),
                safe_float_convert(sex, min_val=0, max_val=1, param_name="Sex"),
                safe_float_convert(cp, min_val=0, max_val=3, param_name="Chest Pain"),
                safe_float_convert(trestbps, min_val=94, max_val=200, param_name="Resting BP"),
                safe_float_convert(chol, min_val=126, max_val=564, param_name="Cholesterol"),
                safe_float_convert(fbs, min_val=0, max_val=1, param_name="Fasting Blood Sugar"),
                safe_float_convert(restecg, min_val=0, max_val=2, param_name="Resting ECG"),
                safe_float_convert(thalach, min_val=71, max_val=202, param_name="Max Heart Rate"),
                safe_float_convert(exang, min_val=0, max_val=1, param_name="Exercise Angina"),
                safe_float_convert(oldpeak, min_val=0, max_val=6.2, param_name="ST Depression"),
                safe_float_convert(slope, min_val=0, max_val=2, param_name="Slope"),
                safe_float_convert(ca, min_val=0, max_val=4, param_name="Colored Vessels"),
                safe_float_convert(thal, min_val=0, max_val=2, param_name="Thal")
            ]
            input_array = np.array(user_input).reshape(1, -1)
            prediction = models['heart_disease'].predict(input_array)
            logger.debug(f"Heart disease input: {user_input}, Prediction: {prediction}")
            if prediction[0] == 1:
                st.success('The person is having heart disease')
                report = generate_disease_report("Heart Disease", "positive")
                st.subheader("Diagnosis Report")
                st.markdown(report)
            else:
                st.success('The person does not have any heart disease')
                st.write("No disease detected, hence no report generated.")
        except Exception as e:
            st.error(f"Heart disease prediction failed: {str(e)}")
            logger.error(f"Heart disease prediction error: {str(e)}")

# Parkinson's Prediction
elif selected == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)', '150', help="70-260 Hz")   
        Jitter_percent = st.text_input('MDVP:Jitter(%)', '0.01', help="0-1%")
        RAP = st.text_input('MDVP:RAP', '0.05', help="0-0.1")   
        Shimmer = st.text_input('MDVP:Shimmer', '0.05', help="0-0.1")   
        APQ3 = st.text_input('Shimmer:APQ3', '0.05', help="0-0.1")  
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)', '200', help="100-600 Hz")  
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)', '0.00005', help="0-0.0001")   
        PPQ = st.text_input('MDVP:PPQ', '0.05', help="0-0.1")   
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)', '0.5', help="0-1.5 dB")   
        APQ5 = st.text_input('Shimmer:APQ5', '0.05', help="0-0.1")   
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)', '100', help="60-240 Hz")   
        DDP = st.text_input('Jitter:DDP', '0.1', help="0-0.3")  
        APQ = st.text_input('MDVP:APQ', '0.05', help="0-0.1")   
        DDA = st.text_input('Shimmer:DDA', '0.1', help="0-0.3")   
        NHR = st.text_input('NHR', '0.1', help="0-0.5")  
    with col4:
        HNR = st.text_input('HNR', '20', help="8-33")   
        RPDE = st.text_input('RPDE', '0.5', help="0-1")  
        DFA = st.text_input('DFA', '0.7', help="0.5-1")  
        spread1 = st.text_input('spread1', '-5', help="-10 to -2")   
    with col5:
        spread2 = st.text_input('spread2', '0.2', help="0-0.5")   
        D2 = st.text_input('D2', '2', help="1-3")   
        PPE = st.text_input('PPE', '0.2', help="0-0.5")   
    if st.button("Parkinson's Test Result"):
        try:
            user_input = [
                safe_float_convert(fo, min_val=70, max_val=260, param_name="MDVP:Fo"),
                safe_float_convert(fhi, min_val=100, max_val=600, param_name="MDVP:Fhi"),
                safe_float_convert(flo, min_val=60, max_val=240, param_name="MDVP:Flo"),
                safe_float_convert(Jitter_percent, min_val=0, max_val=1, param_name="Jitter(%)"),
                safe_float_convert(Jitter_Abs, min_val=0, max_val=0.0001, param_name="Jitter(Abs)"),
                safe_float_convert(RAP, min_val=0, max_val=0.1, param_name="RAP"),
                safe_float_convert(PPQ, min_val=0, max_val=0.1, param_name="PPQ"),
                safe_float_convert(DDP, min_val=0, max_val=0.3, param_name="DDP"),
                safe_float_convert(Shimmer, min_val=0, max_val=0.1, param_name="Shimmer"),
                safe_float_convert(Shimmer_dB, min_val=0, max_val=1.5, param_name="Shimmer(dB)"),
                safe_float_convert(APQ3, min_val=0, max_val=0.1, param_name="APQ3"),
                safe_float_convert(APQ5, min_val=0, max_val=0.1, param_name="APQ5"),
                safe_float_convert(APQ, min_val=0, max_val=0.1, param_name="APQ"),
                safe_float_convert(DDA, min_val=0, max_val=0.3, param_name="DDA"),
                safe_float_convert(NHR, min_val=0, max_val=0.5, param_name="NHR"),
                safe_float_convert(HNR, min_val=8, max_val=33, param_name="HNR"),
                safe_float_convert(RPDE, min_val=0, max_val=1, param_name="RPDE"),
                safe_float_convert(DFA, min_val=0.5, max_val=1, param_name="DFA"),
                safe_float_convert(spread1, min_val=-10, max_val=-2, param_name="spread1"),
                safe_float_convert(spread2, min_val=0, max_val=0.5, param_name="spread2"),
                safe_float_convert(D2, min_val=1, max_val=3, param_name="D2"),
                safe_float_convert(PPE, min_val=0, max_val=0.5, param_name="PPE")
            ]
            input_array = np.array(user_input).reshape(1, -1)
            prediction = models['parkinsons'].predict(input_array)
            logger.debug(f"Parkinson's input: {user_input}, Prediction: {prediction}")
            if prediction[0] == 1:
                st.success("The person has Parkinson's disease")
                report = generate_disease_report("Parkinson's Disease", "positive")
                st.subheader("Diagnosis Report")
                st.markdown(report)
            else:
                st.success("The person does not have Parkinson's disease")
                st.write("No disease detected, hence no report generated.")
        except Exception as e:
            st.error(f"Parkinson's prediction failed: {str(e)}")
            logger.error(f"Parkinson's prediction error: {str(e)}")

# Breast Cancer Prediction
elif selected == "Breast Cancer Prediction":
    st.title("Breast Cancer Prediction")
    st.write("Predicts benign or malignant tumor based on measurements.")
    col1, col2, col3 = st.columns(3)
    with col1:
        radius_mean = st.text_input('Radius Mean', '15', help="6-28")   
        texture_mean = st.text_input('Texture Mean', '20', help="9-39")  
        perimeter_mean = st.text_input('Perimeter Mean', '100', help="43-188")  
        area_mean = st.text_input('Area Mean', '1000', help="143-2501")  
    with col2:
        smoothness_mean = st.text_input('Smoothness Mean', '0.1', help="0.05-0.16")   
        compactness_mean = st.text_input('Compactness Mean', '0.1', help="0.02-0.35") 
        concavity_mean = st.text_input('Concavity Mean', '0.1', help="0-0.43")   
        concave_points_mean = st.text_input('Concave Points Mean', '0.1', help="0-0.2")   
    with col3:
        symmetry_mean = st.text_input('Symmetry Mean', '0.2', help="0.11-0.3")  
        fractal_dimension_mean = st.text_input('Fractal Dimension Mean', '0.07', help="0.05-0.1")   

    st.subheader("Worst Case Features")
    col1, col2, col3 = st.columns(3)
    with col1:
        radius_worst = st.text_input('Radius Worst', '18', help="7-36")   
        texture_worst = st.text_input('Texture Worst', '25', help="12-49")  
        perimeter_worst = st.text_input('Perimeter Worst', '120', help="50-251")   
        area_worst = st.text_input('Area Worst', '1500', help="185-4254")   
    with col2:
        smoothness_worst = st.text_input('Smoothness Worst', '0.15', help="0.07-0.22")   
        compactness_worst = st.text_input('Compactness Worst', '0.2', help="0.03-1.06")   
        concavity_worst = st.text_input('Concavity Worst', '0.2', help="0-1.25")   
        concave_points_worst = st.text_input('Concave Points Worst', '0.15', help="0-0.29")   
    with col3:
        symmetry_worst = st.text_input('Symmetry Worst', '0.3', help="0.16-0.66")  
        fractal_dimension_worst = st.text_input('Fractal Dimension Worst', '0.1', help="0.06-0.21")   

    if st.button("Breast Cancer Test Result"):
        try:
            user_input = [
                safe_float_convert(radius_mean, min_val=6, max_val=28, param_name="Radius Mean"),
                safe_float_convert(texture_mean, min_val=9, max_val=39, param_name="Texture Mean"),
                safe_float_convert(perimeter_mean, min_val=43, max_val=188, param_name="Perimeter Mean"),
                safe_float_convert(area_mean, min_val=143, max_val=2501, param_name="Area Mean"),
                safe_float_convert(smoothness_mean, min_val=0.05, max_val=0.16, param_name="Smoothness Mean"),
                safe_float_convert(compactness_mean, min_val=0.02, max_val=0.35, param_name="Compactness Mean"),
                safe_float_convert(concavity_mean, min_val=0, max_val=0.43, param_name="Concavity Mean"),
                safe_float_convert(concave_points_mean, min_val=0, max_val=0.2, param_name="Concave Points Mean"),
                safe_float_convert(symmetry_mean, min_val=0.11, max_val=0.3, param_name="Symmetry Mean"),
                safe_float_convert(fractal_dimension_mean, min_val=0.05, max_val=0.1, param_name="Fractal Dim Mean"),
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # SE features as placeholders
                safe_float_convert(radius_worst, min_val=7, max_val=36, param_name="Radius Worst"),
                safe_float_convert(texture_worst, min_val=12, max_val=49, param_name="Texture Worst"),
                safe_float_convert(perimeter_worst, min_val=50, max_val=251, param_name="Perimeter Worst"),
                safe_float_convert(area_worst, min_val=185, max_val=4254, param_name="Area Worst"),
                safe_float_convert(smoothness_worst, min_val=0.07, max_val=0.22, param_name="Smoothness Worst"),
                safe_float_convert(compactness_worst, min_val=0.03, max_val=1.06, param_name="Compactness Worst"),
                safe_float_convert(concavity_worst, min_val=0, max_val=1.25, param_name="Concavity Worst"),
                safe_float_convert(concave_points_worst, min_val=0, max_val=0.29, param_name="Concave Points Worst"),
                safe_float_convert(symmetry_worst, min_val=0.16, max_val=0.66, param_name="Symmetry Worst"),
                safe_float_convert(fractal_dimension_worst, min_val=0.06, max_val=0.21, param_name="Fractal Dim Worst")
            ]
            input_array = np.array(user_input).reshape(1, -1)
            standardized_input = scalers['breast_cancer'].transform(input_array)
            prediction = models['breast_cancer'].predict(standardized_input)
            prediction_proba = models['breast_cancer'].predict_proba(standardized_input)
            logger.debug(f"Breast cancer input: {user_input}, Prediction: {prediction}")
            if prediction[0] == 1:
                st.error('The tumor is likely malignant (cancerous)')
                report = generate_disease_report("Breast Cancer", "positive")
                st.subheader("Diagnosis Report")
                st.markdown(report)
            else:
                st.success('The tumor is likely benign (non-cancerous)')
                st.write("No disease detected, hence no report generated.")
            st.write(f"Confidence: {prediction_proba[0][1]:.2%} likelihood of malignancy")
        except Exception as e:
            st.error(f"Breast cancer prediction failed: {str(e)}")
            logger.error(f"Breast cancer prediction error: {str(e)}")

# Tuberculosis Prediction
elif selected == "Tuberculosis Prediction":
    st.title("Tuberculosis Prediction")
    col1, col2 = st.columns(2)
    with col1:
        reporting_area = st.text_input('Reporting Area (numeric ID)', '50', help="0-100")   
        mmwr_year = st.text_input('MMWR Year', '2023', help="2000-2025")   
        mmwr_quarter = st.text_input('MMWR Quarter (1-4)', '2', help="1-4")   
        prev_4q_min = st.text_input('Previous 4 Quarters Min Cases', '10', help="0-1000")   
    with col2:
        prev_4q_max = st.text_input('Previous 4 Quarters Max Cases', '20', help="0-1000")  
        cum_2018 = st.text_input('Cumulative Cases (2018)', '100', help="0-10000")   
        cum_2017 = st.text_input('Cumulative Cases (2017)', '100', help="0-10000")   

    if st.button("Predict Tuberculosis Cases"):
        try:
            user_input = [
                safe_float_convert(reporting_area, min_val=0, max_val=100, param_name="Reporting Area"),
                safe_float_convert(mmwr_year, min_val=2000, max_val=2025, param_name="MMWR Year"),
                safe_float_convert(mmwr_quarter, min_val=1, max_val=4, param_name="MMWR Quarter"),
                safe_float_convert(prev_4q_min, min_val=0, max_val=1000, param_name="Prev 4Q Min"),
                safe_float_convert(prev_4q_max, min_val=0, max_val=1000, param_name="Prev 4Q Max"),
                safe_float_convert(cum_2018, min_val=0, max_val=10000, param_name="Cum 2018"),
                safe_float_convert(cum_2017, min_val=0, max_val=10000, param_name="Cum 2017")
            ]
            input_array = np.array(user_input).reshape(1, -1)
            standardized_input = scalers['tuberculosis'].transform(input_array)
            prediction = models['tuberculosis'].predict(standardized_input)
            logger.debug(f"Tuberculosis input: {user_input}, Prediction: {prediction}")
            tb_diagnosis = f"Predicted Tuberculosis Cases: {prediction[0]:.2f}"
            if prediction[0] > 0:
                st.success(tb_diagnosis)
                report = generate_disease_report("Tuberculosis", "positive")
                st.subheader("Diagnosis Report")
                st.markdown(report)
            else:
                st.success(tb_diagnosis)
                st.write("No significant risk detected, hence no report generated.")
        except Exception as e:
            st.error(f"Tuberculosis prediction failed: {str(e)}")
            logger.error(f"Tuberculosis prediction error: {str(e)}")

# Liver Disease Prediction
elif selected == 'Liver Disease Prediction':
    st.title('Liver Disease Prediction')
    col1, col2, col3 = st.columns(3)
    with col1:
        tot_bilirubin = st.text_input('Total Bilirubin', '0.5', help="0.1-50 mg/dL")   
        direct_bilirubin = st.text_input('Direct Bilirubin', '0.1', help="0-20 mg/dL")   
        sgpt = st.text_input('SGPT (Alanine Aminotransferase)', '10', help="7-56 U/L")  
    with col2:
        sgot = st.text_input('SGOT (Aspartate Aminotransferase)', '10', help="5-40 U/L")   
        alkphos = st.text_input('Alkaline Phosphatase', '50', help="44-147 U/L")  
        albumin = st.text_input('Albumin', '4.0', help="3.5-5.5 g/dL")   
    with col3:
        ag_ratio = st.text_input('A/G Ratio', '1.2', help="1-2.5")   
        tot_proteins = st.text_input('Total Proteins', '7.0', help="6-8.5 g/dL")   

    if st.button('Liver Disease Test Result'):
        try:
            user_input = [
                safe_float_convert(tot_bilirubin, min_val=0.1, max_val=50, param_name="Total Bilirubin"),
                safe_float_convert(direct_bilirubin, min_val=0, max_val=20, param_name="Direct Bilirubin"),
                safe_float_convert(sgpt, min_val=7, max_val=56, param_name="SGPT"),
                safe_float_convert(sgot, min_val=5, max_val=40, param_name="SGOT"),
                safe_float_convert(alkphos, min_val=44, max_val=147, param_name="Alkaline Phosphatase"),
                safe_float_convert(albumin, min_val=3.5, max_val=5.5, param_name="Albumin"),
                safe_float_convert(ag_ratio, min_val=1, max_val=2.5, param_name="A/G Ratio"),
                safe_float_convert(tot_proteins, min_val=6, max_val=8.5, param_name="Total Proteins")
            ]
            input_array = np.array(user_input).reshape(1, -1)
            imputer = SimpleImputer(strategy='median')
            input_imputed = imputer.fit_transform(input_array)
            standardized_input = scalers['liver'].transform(input_imputed)
            prediction = models['liver'].predict(standardized_input)
            logger.debug(f"Liver input: {user_input}, Prediction: {prediction}")
            if prediction[0] == 1:
                st.success('The person is likely to have liver disease')
                report = generate_disease_report("Liver Disease", "positive")
                st.subheader("Diagnosis Report")
                st.markdown(report)
            else:
                st.success('The person is not likely to have liver disease')
                st.write("No disease detected, hence no report generated.")
        except Exception as e:
            st.error(f"Liver disease prediction failed: {str(e)}")
            logger.error(f"Liver disease prediction error: {str(e)}")

# Stroke Prediction
elif selected == 'Stroke Prediction':
    st.title('Stroke Prediction')
    st.write("Enter patient information to predict stroke risk.")
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
        age = st.number_input('Age', min_value=0, max_value=120, value=30)  # 
        hypertension = st.selectbox('Hypertension', ['No', 'Yes'])
    with col2:
        heart_disease = st.selectbox('Heart Disease', ['No', 'Yes'])
        ever_married = st.selectbox('Ever Married', ['No', 'Yes'])
        work_type = st.selectbox('Work Type', ['Private', 'Self-employed', 'Govt job',
                                              'Children', 'Never worked'])
    with col3:
        residence_type = st.selectbox('Residence Type', ['Urban', 'Rural'])
        avg_glucose_level = st.number_input('Average Glucose Level (mg/dL)', min_value=0.0, value=80.0)   
        bmi = st.number_input('BMI', min_value=10.0, max_value=60.0, value=25.0)   
        smoking_status = st.selectbox('Smoking Status', ['never smoked', 'formerly smoked', 'smokes', 'Unknown'])

    if st.button('Stroke Risk Assessment'):
        try:
            gender_encoded = {'Male': 0, 'Female': 1, 'Other': 2}.get(gender, 0)
            hypertension_encoded = 1 if hypertension == 'Yes' else 0
            heart_disease_encoded = 1 if heart_disease == 'Yes' else 0
            ever_married_encoded = 1 if ever_married == 'Yes' else 0
            work_type_mapping = {'Private': 0, 'Self-employed': 1, 'Govt job': 2, 'Children': 3, 'Never worked': 4}
            work_type_encoded = work_type_mapping.get(work_type, 0)
            residence_type_encoded = 1 if residence_type == 'Urban' else 0
            smoking_status_mapping = {'never smoked': 0, 'formerly smoked': 1, 'smokes': 2, 'Unknown': 3}
            smoking_status_encoded = smoking_status_mapping.get(smoking_status, 3)

            numerical_features = np.array([[age, avg_glucose_level, bmi]])
            scaled_numerical = scalers['stroke'].transform(numerical_features)[0]
            user_input = [
                gender_encoded,
                scaled_numerical[0],   
                hypertension_encoded,
                heart_disease_encoded,
                ever_married_encoded,
                work_type_encoded,
                residence_type_encoded,
                scaled_numerical[1],  
                scaled_numerical[2],   
                smoking_status_encoded
            ]
            input_array = np.array(user_input).reshape(1, -1)
            prediction = models['stroke'].predict(input_array)
            prediction_proba = models['stroke'].predict_proba(input_array)
            logger.debug(f"Stroke input: {user_input}, Prediction: {prediction}")
            if prediction[0] == 1:
                st.error('High risk of stroke detected')
                report = generate_disease_report("Stroke", "positive")
                st.subheader("Diagnosis Report")
                st.markdown(report)
            else:
                st.success('Low risk of stroke detected')
                st.write("No disease detected, hence no report generated.")
            st.write(f"Probability of stroke risk: {prediction_proba[0][1]:.2%}")
        except Exception as e:
            st.error(f"Stroke prediction failed: {str(e)}")
            logger.error(f"Stroke prediction error: {str(e)}")