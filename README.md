# Association Rule Learning
Association Rule Learning based application from recommender system 🤞 

### 1. Data Preprocessing
### 2. Preparing the ARL Data Structure (Invoice-Product Matrix)
### 3. Issuance of Association Rules
### 4. Preparing the Script of the Work
### 5. Making Product Recommendations to Users in the Cart Stage

# 1. Data Preprocessing

--> When people add products to their carts, which products should I recommend to these people ⁉

--> The difficult part is to bring the data to the special data format that ARL expects from Apriori, this is the most valuable part of the project.

--> This is the main difficulty that can be encountered in real life.


NOTE: If an error occurs while reading the file from Excel, the following steps should be followed;
!pip install openpyxl is downloaded
next;

df_ = pd.read_excel(r"dataset\online_retail_II.xlsx", sheet_name="Year 2010-2011", engine="openpyxl")
It is sufficient to add the engine="openpyxl" parameter to the read_excel function.

-->  You can download the data from the link here.   🕊     --> https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

To use the apriori algorithm, the mlxtend library must be imported.

### Classic data preprocessing operations performed
  1) describe was looked at and outlier thresholding was done
  2) Values in quantity and price variables have been destroyed
  --> One of the situations that reveals negative values is the C statement at the beginning of Invoices. These represent returns. Since returns are processed with - values have arrived

3) NA values were detected, NA rows were deleted because the data set was rich

# 2. Preparing the ARL Data Structure (Invoice-Product Matrix)


There should be invoices in the rows and products in the columns. Whether a product is on the invoice or not is expressed as 0-1.

The data set is reduced to a specific country

# 3. Issuance of Association Rules

--> antecedents: frequency of occurrence of first product <br/>
--> consequences: frequency of occurrence of second product  <br/>
--> antecedent support: probability of observing the first product  <br/>
--> consequent support: probability of observing the second product  <br/>
--> support: is the probability of two given products appearing together  <br/>
--> confidence: probability of purchasing y since product x is purchased  <br/>
--> lift: When product x is purchased, the probability of purchasing product y increases by 17 times  <br/>
--> leverage: leverage effect, it is a value similar to lift, but the leverage value tends to prioritize values with high support, so it has a slight bias.  <br/>
--> lift value, although less frequent, can capture some relationships, so it is an unbiased and more valuable metric for us  <br/>
--> conviction: is the expected frequency of product x without product y, or expected frequency of product y without product x  <br/>

# 4. Preparing the Script of the Work


# 5. Making Product Recommendations to Users in the Cart Stage


 Example:
 User sample product id: 22492
 User added this product to the cart

--> Who or which product can be recommended against possible scenarios that may have occurred before is kept in the tables, and the information available as soon as the user logs in and adds a product to his cart is returned from the databases.


IN SUMMARY:

ARL, one of the recommendation systems, was examined. Apriori algorithm was used. A recommendation was made to a customer who had added an item to his cart.

You can direct your work in the field of data science by taking inspiration from this study.
Goodbye 👋
















