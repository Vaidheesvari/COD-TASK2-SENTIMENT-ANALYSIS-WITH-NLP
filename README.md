# COD-TASK2-SENTIMENT-ANALYSIS-WITH-NLP

*COMPANY*:COD TECH IT SOLUTIONS

*NAME*:VAIDHEESVARI.M.K

*INTERN ID*:CT06DG2683

*DOMAIN*:MACHINE LEARNING

*DURATION*:8 WEEKS

*MENTOR*:NEELA SANTHOSH

*DESCRIPTION OF THE TASK*:

This project involves building a sentiment analysis model that classifies customer reviews as either positive or negative. Sentiment analysis, a subset of Natural Language Processing (NLP), is widely used in areas such as product feedback, social media monitoring, and customer support automation. In this task, the main objective is to preprocess textual data, convert it into numerical form using TF-IDF vectorization, and then train a Logistic Regression model to predict the sentiment of unseen reviews.

The process begins by collecting or using a dataset of customer reviews, each labeled with its corresponding sentiment. The dataset typically consists of two columns: one for the review text and the other for the sentiment label. Before building the model, it's essential to clean and preprocess the raw text to remove unnecessary elements such as HTML tags, URLs, digits, and punctuation. The text is also converted to lowercase to maintain consistency and reduce dimensionality.

After cleaning the text, the next step is vectorization. Since machine learning models cannot work directly with text data, the words must be transformed into numerical features. This is done using TF-IDF (Term Frequency-Inverse Document Frequency), which assigns importance to words based on how often they appear in a document relative to their appearance in all documents. This ensures that common but less informative words like "the" or "is" have lower weights, while rare but meaningful words have higher weights.

Once the textual data is vectorized, it is split into training and testing sets. The training data is used to teach the model how to recognize patterns in positive and negative reviews, while the test set evaluates its performance on unseen data. Logistic Regression, a linear classifier, is used as the model because of its simplicity, speed, and effectiveness in binary classification tasks like sentiment analysis.

After training, the model’s performance is assessed using evaluation metrics such as accuracy, precision, recall, and F1-score. A confusion matrix is also generated to visually inspect how many predictions were correct and where the model made mistakes.

The entire project is implemented in Python using a Jupyter Notebook within Visual Studio Code (VS Code). Libraries such as pandas are used for data manipulation, scikit-learn for machine learning tasks, and matplotlib or seaborn for visualization. The notebook is structured into clear steps: data loading, preprocessing, vectorization, model training, and evaluation, ensuring transparency and reproducibility.

This project demonstrates how simple yet powerful machine learning techniques can be used to extract insights from text data. The resulting model can be used to automatically classify new reviews, helping businesses monitor customer feedback, improve service quality, and make informed decisions based on sentiment trends.

In summary, the task of sentiment analysis using TF-IDF and Logistic Regression provides a practical introduction to text classification, NLP workflows, and machine learning model evaluation — all crucial skills for data science and AI development.

*OUTPUT*:

<img width="702" height="318" alt="Image" src="https://github.com/user-attachments/assets/9e687a2b-af27-4a61-94c9-c9627ac100fa" />

<img width="935" height="702" alt="Image" src="https://github.com/user-attachments/assets/aa3b83a2-6044-4627-b97f-3bb59f70bd67" />
