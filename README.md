# Big-Data-Predictive-Analytics-for-NBA-Player-Evaluation-Using-AI-Modeling
Analyzing NBA player performance and salaries using big data analytics and machine learning. It leverages data preprocessing, statistical analysis, and predictive modeling to provide insights into player metrics, salary allocation, and their relationship with game outcomes. It includes position classification, salary prediction, and evaluation.


---



#### Objectives  
- Analyze the influence of performance metrics (points, assists, rebounds) on NBA player salaries.  
- Predict player positions based on their performance using machine learning models.  
- Evaluate the efficiency of individual players and teams.  
- Identify patterns and trends in player scoring and performance metrics.  
- Develop interactive visualizations and dashboards for comprehensive insights.

---


### Data Understanding

### 1. **Dataset Overview**  
- **Player Stats Dataset**: 
  - Contains **14,573 entries** and **31 columns**, including various performance metrics such as points (PTS), assists (AST), and rebounds (REB).
  - Covers multiple seasons and includes detailed player performance metrics for each game.

- **Salaries Dataset**: 
  - Contains **9,456 entries** and **6 columns**, including player name, position, team, salary, and season information.
  - Focuses on salary information tied to players’ contracts and teams.

<img width="600" alt="Screenshot 2025-01-02 at 5 17 05 PM" src="https://github.com/user-attachments/assets/ff0756b0-a2a0-4262-a0cc-2514356c566e" />


---

### 2. **Initial Exploration**  
- **Dimensions**:
  - The datasets' dimensions confirm their sizes: Player Stats Dataset (14,573 rows × 31 columns), Salaries Dataset (9,456 rows × 6 columns).
- **Null Values**:
  - Preliminary analysis identified columns with missing values, including field goal percentage (FG%), assists (AST), and efficiency ratings (EFF). These were handled during the **Data Preparation** phase.
- **Duplicate Entries**:
  - Both datasets were checked for duplicate rows, ensuring data integrity for further analysis.
<img width="500" alt="Screenshot 2025-01-02 at 5 24 30 PM" src="https://github.com/user-attachments/assets/8642faba-c71f-4ca4-9070-8e657d0c2178" />

  - The graph shows the distribution of points (PTS) scored by players, as visualized by a histogram with a kernel density estimate (KDE) overlaid.

<img width="500" alt="Screenshot 2025-01-02 at 5 24 54 PM" src="https://github.com/user-attachments/assets/0e6bcb25-04cb-43e7-b0db-d57c4b84f147" />

  - The graph illustrates the distribution of Field Goal Percentage (FG%) among players, using a histogram with an overlaid kernel density estimate (KDE).
   
<img width="500" alt="Screenshot 2025-01-02 at 5 25 22 PM" src="https://github.com/user-attachments/assets/90562eeb-0999-4c33-980a-0515410547d3" />

  - The graph shows the distribution of Minutes Played (MP) by players, represented by a histogram with a kernel density estimate (KDE) overlay.

<img width="500" alt="Screenshot 2025-01-02 at 5 27 24 PM" src="https://github.com/user-attachments/assets/4b986fca-9d24-4bba-8e30-2ba058a45ae2" />

  - The box plot provides a visual summary of NBA player salaries, showing the distribution, central tendency, and presence of outliers.

<img width="500" alt="Screenshot 2025-01-02 at 5 27 43 PM" src="https://github.com/user-attachments/assets/9c0c1627-8b0c-4623-b338-a08e647ee06d" />

  - The scatter plot visualizes the relationship between Minutes Played (MP) and Points Scored (PTS) by players.

<img width="500" alt="Screenshot 2025-01-02 at 5 28 07 PM" src="https://github.com/user-attachments/assets/5127a2e3-9985-401d-b5d1-04bb7519f712" />

  - The scatter plot shows the distribution of NBA player salaries over different seasons from 2000 to 2020.
---



### 3. **Data Preparation**  
- **Data Cleaning**:
  - Handled missing values in columns like field goal percentages, assists, and rebounds by calculating averages or replacing them with zeros when necessary.
  - Standardized team names and merged datasets (player stats and salaries) using key identifiers like player names and seasons.
    <img width="700" alt="Screenshot 2025-01-02 at 5 35 17 PM" src="https://github.com/user-attachments/assets/7682b225-dbb5-402c-9cb2-613b354715e4" />

---

### 4. **Exploratory Data Analysis (EDA)**  
- **Visualizations**:
  - Histograms, scatter plots, and correlation heatmaps to uncover patterns and relationships.
  - Salary trends by season and performance metrics across positions.
    ## A. Predictive Analytics and Performance Metrics for NBA Player Position Classification
    <img width="500" alt="Screenshot 2025-01-02 at 8 29 00 PM" src="https://github.com/user-attachments/assets/304e1260-4b18-4eb4-9bb6-60d5eb3ee506" />
    <img width="500" alt="Screenshot 2025-01-02 at 8 30 50 PM" src="https://github.com/user-attachments/assets/7279983b-812a-44b8-88f6-c4aaa0308aa8" />
    <img width="500" alt="Screenshot 2025-01-02 at 8 31 02 PM" src="https://github.com/user-attachments/assets/eccfaa6b-15c2-44c1-99c4-231e35bd32a2" />
    <img width="500" alt="Screenshot 2025-01-02 at 8 31 11 PM" src="https://github.com/user-attachments/assets/52e6512c-291b-4a1f-94a3-d5d5fc3e4e9e" />
    <img width="500" alt="Screenshot 2025-01-02 at 8 31 32 PM" src="https://github.com/user-attachments/assets/1d9d8c0b-d668-462f-a93d-d6de94a2a19d" />
    <img width="500" alt="Screenshot 2025-01-02 at 8 31 53 PM" src="https://github.com/user-attachments/assets/d217a821-0e82-47ec-b296-2917e981d7e6" />
    <img width="500" alt="Screenshot 2025-01-02 at 8 32 04 PM" src="https://github.com/user-attachments/assets/b7f37d52-d03c-4496-92ba-fb6ed49ef5be" />
    <img width="500" alt="Screenshot 2025-01-02 at 8 32 19 PM" src="https://github.com/user-attachments/assets/5ca2d7b3-34a9-438b-9077-7d830785f46d" />
    <img width="500" alt="Screenshot 2025-01-02 at 8 32 41 PM" src="https://github.com/user-attachments/assets/234b4956-9199-451c-aca1-c4cb6ef57d70" />
    <img width="500" alt="Screenshot 2025-01-02 at 8 32 55 PM" src="https://github.com/user-attachments/assets/38278f08-fb06-4bbc-9045-aff85731a986" />
    <img width="500" alt="Screenshot 2025-01-02 at 8 33 10 PM" src="https://github.com/user-attachments/assets/b349ee2b-7a29-455d-8f38-10a3396d08cc" />
    <img width="500" alt="Screenshot 2025-01-02 at 8 33 35 PM" src="https://github.com/user-attachments/assets/0f8c24a1-589d-4918-91bf-ecd762bbc7bf" />
    <img width="500" alt="Screenshot 2025-01-02 at 8 33 46 PM" src="https://github.com/user-attachments/assets/ee0299da-8e87-42c9-84c6-07aff15fc93b" />
    <img width="500" alt="Screenshot 2025-01-02 at 8 33 56 PM" src="https://github.com/user-attachments/assets/0b906c03-f20d-49b4-925d-9a3e96c22167" />
    <img width="500" alt="Screenshot 2025-01-02 at 8 34 07 PM" src="https://github.com/user-attachments/assets/9ebc80ac-7cc1-4b12-8116-9ed506e8fdb2" />
    <img width="500" alt="Screenshot 2025-01-02 at 8 34 18 PM" src="https://github.com/user-attachments/assets/724d9f3f-ff17-49e2-b3dd-43157f7619d6" />
 
    
    These scatter plots illustrate the relationship between player salaries and key performance metrics (such as points, assists, and rebounds) across various NBA positions, including Small     Forward (SF), Point Guard (PG), Power Forward (PF), and others.

    #### Top Earners by Position
    <img width="600" alt="Screenshot 2025-01-02 at 8 35 44 PM" src="https://github.com/user-attachments/assets/b86800b8-2196-4713-876e-eb3e8513e051" />
    
    This section creates a bar chart that highlights the top 5 highest-paid players within each position. It visually compares the salaries of the top earners across different roles on the       basketball court.

    #### Average Salary by Position:
    <img width="600" alt="Screenshot 2025-01-02 at 8 36 48 PM" src="https://github.com/user-attachments/assets/d2268d23-28d7-4564-9fef-03f85b9d94f0" />

    Here, the code produces a bar chart showing the average salary for each position. This helps to identify which positions generally command higher or lower salaries in the NBA.


    #### Performance Distribution within Positions (Boxplots):
    <img width="300" alt="Screenshot 2025-01-02 at 8 40 16 PM" src="https://github.com/user-attachments/assets/f7c92306-0e47-436d-bd84-7cdbf14ee0a4" />
    <img width="300" alt="Screenshot 2025-01-02 at 8 40 41 PM" src="https://github.com/user-attachments/assets/5b9daf75-485f-4030-b5a3-a58c8c8f8122" />
    <img width="300" alt="Screenshot 2025-01-02 at 8 40 50 PM" src="https://github.com/user-attachments/assets/9d20711d-f767-49b4-b2b5-fa70e73e2a2f" />
    <img width="300" alt="Screenshot 2025-01-02 at 8 41 11 PM" src="https://github.com/user-attachments/assets/4788421b-3e80-4090-8ef1-a6dc717f1fb3" />

    Finally, the code creates boxplots that show the distribution of key performance metrics (like points, assists, rebounds, and field goals) within each position. This visualization helps     to understand the range and median of these metrics for different player roles.

    <img width="500" alt="Screenshot 2025-01-02 at 8 43 06 PM" src="https://github.com/user-attachments/assets/92607df8-b7ab-4b2b-bdc3-4df874d519db" />
    
    This confusion matrix indicates the overall performance of the location classification model in which it maps the actual post of the players on the vertical axis and the predicted post      on the horizontal axis. As can be observed from the matrix, each cell indicates the occurrences of matches for actual and predicted positions.


  ## B. Correlation Analysis of Performance Metrics and Salaries
  ### Correlation Visualization
  
  <img width="700" alt="Screenshot 2025-01-02 at 9 09 08 PM" src="https://github.com/user-attachments/assets/d8b2f849-8322-4ea6-ae5b-4a7778d6973c" />

  ### Distrubution Analysis
  <img width="700" alt="Screenshot 2025-01-02 at 9 14 28 PM" src="https://github.com/user-attachments/assets/2437066e-6c56-4538-9a2c-d4cd2e56ba23" />

  ### Outlier Detection, Removal and Visualization
  The box plot below represents the distribution of key performance metrics (e.g., PTS, FG, FGA, FT, etc.) in your NBA dataset before and after outlier removal.
  First Set of Box Plots (Before Outlier Removal)

  #### First Set of Box Plots (Before Outlier Removal)
  
  <img width="700" alt="Screenshot 2025-01-02 at 9 15 15 PM" src="https://github.com/user-attachments/assets/a4385790-3544-4157-8ac3-dd247b10f1b9" />

  In the first set, the box plots show the distribution of each performance metric, with a noticeable number of outliers (indicated by the dots beyond the whiskers). These outliers are        common in datasets involving sports performance, where certain players significantly outperform others. The presence of these outliers suggests high variability in player performance        across different metrics.


  #### Second Set of Box Plots (After Outlier Removal)

  <img width="700" alt="Screenshot 2025-01-02 at 9 16 46 PM" src="https://github.com/user-attachments/assets/83150561-4138-48f3-b500-55090a99469f" />

  The second set of box plots illustrates the same metrics but after outliers have been removed using the Interquartile Range (IQR) method. Here, the range of values is more constrained,      and there are fewer outliers, indicating a more normalized dataset. Removing outliers helps in achieving more accurate predictive modeling by reducing the influence of extreme values that   could skew the results.

  ## C. Analysis of Age against Various Performance Matrices
  
  ### Data Distribution
  <img width="700" alt="image" src="https://github.com/user-attachments/assets/1e19eab1-2a95-4052-913d-eaf576965efa" />
  


  Overall, these visualizations help to understand how the data was cleaned and prepared for analysis, ensuring that the predictive models built using this data will be more reliable.

  
---

### 5. **Predictive Analysis**  
  <img width="700" alt="Screenshot 2025-01-02 at 9 23 11 PM" src="https://github.com/user-attachments/assets/e5409151-21cc-4b3e-92c6-9b8fb14294bb" />

  The feature importance chart illustrates the relative significance of each feature in predicting NBA player salaries using the Random Forest model.

  <img width="700" alt="Screenshot 2025-01-02 at 9 23 48 PM" src="https://github.com/user-attachments/assets/dab5124f-f806-45b0-b567-a064ded59be7" />

  The scatter plot above compares the actual versus predicted salaries for NBA players, with the dashed line representing the ideal scenario where predictions perfectly match the actual       values.

  <img width="700" alt="Screenshot 2025-01-02 at 9 24 42 PM" src="https://github.com/user-attachments/assets/bf169f98-f527-4d30-bc6c-52909fd50576" />

  The distribution of residuals plot shows the difference between the predicted and actual salaries for the NBA players. Ideally, the residuals should be normally distributed around zero,     indicating that the model's predictions are unbiased, and errors are randomly distributed.

  <img width="700" alt="Screenshot 2025-01-02 at 9 26 13 PM" src="https://github.com/user-attachments/assets/a80665fe-d997-4fb9-9d16-560a78fefc9f" />

  The learning curves plot illustrates the performance of the model as the training size increases, using R-squared as the evaluation metric. Here's what the plot reveals:
  
**Training Score (Red Line):** The training score remains consistently high, around 0.85, indicating that the model is fitting the training data well. This high and stable training score suggests that the model has sufficient capacity to learn the training data effectively.
  
**Cross-Validation Score (Green Line):** The cross-validation score, which represents the model's performance on unseen data, starts lower (around 0.3) and gradually increases as the training size grows, stabilizing near 0.45. This gap between the training score and the cross-validation score indicates that the model is overfitting—performing well on the training data but not generalizing as effectively to new data.
  
**Trend Analysis:** The cross-validation score improves as the training size increases, which is a positive sign. However, the score doesn't increase much beyond a certain point, suggesting that merely adding more data might not significantly improve the model's generalization ability.


---

### Challenges and Future Enhancements  
- **Challenges**:
  - Overlapping roles and outliers complicate predictions.
  - High variance in salaries due to factors outside performance metrics.  
- **Future Enhancements**:
  1. Integrate deep learning models (e.g., LSTMs, BERT) for better predictions.
  2. Expand datasets to include additional attributes like injuries or playoff performance.
  3. Refine position classification using ensemble techniques or advanced feature engineering.
  4. Improve dashboards with real-time data integration for dynamic analysis.

---
