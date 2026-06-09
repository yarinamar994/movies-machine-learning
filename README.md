# פרויקט חיזוי ציוני סרטים (IMDb)

## סקירת הפרויקט
פרויקט זה נועד לבנות מודל למידת מכונה המסוגל לחזות את הציון הממוצע (`averageRating`) של סרטים באתר IMDb, אך ורק על בסיס מידע הזמין **טרם יציאת הסרט לאקרנים**. 

הפרויקט כולל שלבי טעינת נתונים, ניקוי, הנדסת מאפיינים חכמה (Feature Engineering), בניית Pipelines למניעת זליגת נתונים (Data Leakage), אימון שני מודלים מרכזיים (Elastic Net ו-Random Forest), חיפוש היפר-פרמטרים (Grid Search) והערכת ביצועים באמצעות 10-Fold Cross Validation. כמו כן, בוצעו ניתוחי חריגים (Outliers), בדיקות מובהקות סטטיסטית (Levene's Test) וניתוחי הוגנות (Fairness Analysis).

## קבצים בפרויקט
* `movie_rating_model.ipynb` – מחברת Jupyter המכילה את כל הקוד מתחילתו ועד סופו.
* `movie_rating_pipeline.pkl` – המודל הסופי (Elastic Net) השמור יחד עם ה-Pipeline המלא שלו, מוכן לשימוש.
* `requirements.txt` – רשימת הספריות והגרסאות המדויקות הנדרשות להרצת הפרויקט.
* `README.md` – קובץ הוראות זה.

## הכנת הנתונים (Data Preparation)
הפונקציה `prepare_data(df)` אחראית על עיבוד נתוני הגלם. פונקציה זו נבנתה בגישת **Whitelist** – הגדרת עמודות מורשות בלבד, מה שמבטיח שום מידע עתידי (כמו קופות או מספר מצביעים) לא יוכל לזלוג פנימה, גם אם יתווסף לדאטה-סט בעתיד.

פעולות מרכזיות:
* **המרה וניקוי:** טיפול בשדות טקסט והמרה למספרים.
* **חלוקה לעשורים:** יצירת משתני Dummy לפי עשור היציאה של הסרט במקום שנת יציאה רציפה.
* **עיבוד תקציב חכם:** פונקציה מתקדמת הממירה תקציבים במטבעות שונים, מזהה מילות קידומת, ומנרמלת הכל לדולרים.
* **קידוד שפות ומדינות:** יצירת אינדיקטורים בינאריים למדינות ושפות מובילות בתעשייה, תוך איחוד מדינות/שפות נדירות לקטגוריית "אחר".

## הנדסת מאפיינים מתקדמת וטרנספורמרים
כדי להתמודד עם שדות מורכבים מבלי לגרום לזליגת נתונים בתוך ה-Cross Validation, נבנו מחלקות Transformer ייעודיות המשתלבות ישירות ב-Pipeline של scikit-learn:

1.  `GenreTargetEncoder`: ממצע את הציון לפי ז'אנרים תוך שימוש בהחלקת נתונים (Smoothing) למניעת משקל יתר לז'אנרים נדירים.
2.  `AdvancedCastEncoder`: מנתח את צוות השחקנים ומחשב מדדים מתקדמים (ממוצע, ניסיון קאסט) תוך שימוש בהחלקה.
3.  `RuntimeHighPolynomial`: מתמודד עם ערכים חסרים בזמן הריצה ויוצר מאפיינים פולינומיאליים לתפיסת קשרים לא ליניאריים.
4.  `BudgetLogWithMissingFlag`: מחליף תקציבים חסרים בחציון, יוצר עמודת דגל (`is_budget_missing`) ומפעיל התמרת לוג ($\log(x+1)$) על התקציב.

## המודלים
בפרויקט נבחנו שני מודלים, שלשניהם בוצע כיוונון היפר-פרמטרים באמצעות `GridSearchCV`:

1.  **Elastic Net:** מודל לינארי המשלב רגולריזציות מסוג L1 ו-L2. (מודל זה נבחר לשמירה כקובץ ה-pkl).
2.  **Random Forest Regressor:** מודל מבוסס עצים שבוצע עבורו חיפוש ארכיטקטורה אופטימלית למניעת התאמת-יתר.

## הערכת ביצועים
המודלים הוערכו בקפידה תחת מסגרת של **10-Fold Cross Validation** עם Seed קבוע (42). 

המדדים המדווחים עבור כל מודל:
* **RMSE** (Root Mean Squared Error)
* **MAE** (Mean Absolute Error)
* **R²** (R-squared)

## שימוש במודל השמור
המודל השמור (`model.pkl`) מכיל את כל ה-Pipeline הנדרש. 
לפני השימוש יש לטעון את בלוק 2 ובלוק 3 מהמחברת!!! ללא הפונקציה פריפר דאטה והמחלקות המוגדרות זה לא ירוץ!

דוגמה לשימוש עתידי בנתונים חדשים:

```python
import pandas as pd
import pickle

# טעינת המודל
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# טעינת נתונים חדשים
df_new = pd.read_csv("new_movies.csv")

# ניקוי ראשוני (הכנת העמודות לפי ה-Whitelist)
X_new = prepare_data(df_new)

# קבלת התחזיות
predictions = model.predict(X_new)
חשוב לקרוא!!!!!!!!!!!!!
היי חן ואליאור אצלי בפונקציה פריפר דאטה אני לא מוחק שורות שאין בהם ציון כדי שזה יהיה טוב תעשייתי (כי בתעשייה אני אקבל דאטה שהיא בלי דירוג מן הסתם ואז זה ימחק הכל ) עכשיו לא היה לי ברור אם ציפיתם למחיקת שורות או רק שזה יוציא את הפיצרים מוכנים ללמידה , יש לציין שאני יודע לעשות את שתי המקרים . ככה הטסט צריך להתבצע  :
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

test_data_path = r"C:\Users\yarin\Downloads\dataset.csv" 
model_path = r"C:\Users\yarin\Desktop\פרויקט סרטים\movie_rating_pipeline.pkl" 

print("1. טוען את קובץ המבחן...")
df_test = pd.read_csv(test_data_path, low_memory=False)

if 'averageRating' in df_test.columns:
    print("2. מכין נתונים להשוואה (מוחק שורות ללא ציון מהקובץ)...")
    df_test_clean = df_test.dropna(subset=['averageRating']).copy()
    y_true = df_test_clean['averageRating']
else:
    print("2. לא נמצאה עמודת ציון בקובץ.")
    df_test_clean = df_test.copy()
    y_true = None

print("3. מעביר את הנתונים דרך prepare_data...")
X_test = prepare_data(df_test_clean)

print("4. טוען את המודל מתוך ה-Pickle...")
model = joblib.load(model_path)

print("5. מפיק תחזיות!")
predictions = model.predict(X_test)

if y_true is not None:
    rmse = np.sqrt(mean_squared_error(y_true, predictions))
    r2 = r2_score(y_true, predictions)
    
    print("\n🏆 === ציון המודל === 🏆")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")
else:
    print("\nהנה 15 התחזיות הראשונות:")
    print(np.round(predictions[:15], 2))
