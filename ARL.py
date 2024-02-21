############################################
# ASSOCIATION RULE LEARNING (BİRLİKTELİK KURALI ÖĞRENİMİ)
############################################

# 1. Veri Ön İşleme
# 2. ARL Veri Yapısını Hazırlama (Invoice-Product Matrix)
# 3. Birliktelik Kurallarının Çıkarılması
# 4. Çalışmanın Scriptini Hazırlama
# 5. Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak

############################################
# 1. Veri Ön İşleme
############################################

"""
Kişiler sepetlerine ürün eklediklerinde bu kişilere hangi ürünleri önermeliyim
İşin zor kısmı ARL’in aprior’nin beklediği özel veri formatına veriyi getirmektir, projenin en değerlli kısmı burasıdır.

"""


"""
NOT : Eğer excel'den dosya okumakta hata alınırsa şu adımlar izlenmelidir;
!pip install openpyxl  il indirilir
ardından;

df_ = pd.read_excel(r"C:\Users\MerveATASOY\Desktop\data_scientist_miuul\egitim_teorik_icerikler\Bolum_6_Tavsiye_Sistemleri\dataset\online_retail_II.xlsx",
                    sheet_name="Year 2010-2011", engine="openpyxl")
bu durumda hata çözülebilir
"""

# Bu projenin asıl zorluğu ne fonksiyonları uygulamakta ne de yorumlar yapmaktır,
# Klasik kaynaklardaki toy datasetlerden farklı olan bu veri setini özel veri yapısına dönüştürmektir
# Gerçek hayatta uğraşıldığından asıl karşılaşılacak olan zorluk budur



# !pip install mlxtend
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

df_ = pd.read_excel(r"C:\Users\MerveATASOY\Desktop\data_scientist_miuul\egitim_teorik_icerikler\Bolum_6_Tavsiye_Sistemleri\dataset\online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")
df = df_.copy()
df.head()
# pip install openpyxl
# df_ = pd.read_excel("datasets/online_retail_II.xlsx",
#                     sheet_name="Year 2010-2011", engine="openpyxl")


df.describe().T   #sadece sayılar değişkenleri betimler
df.isnull().sum()
df.shape

#öncelikle klasik veri ön işleme işlemleri gerçekleştirilecek
# 1)describe'na bakıldığında aykırı değerler olduğu görülmektedir
# 2) quantity ve price'larda eksi değerler vardır, olmamalı
# eksi değerleri ortaya çıkaran durumlardan birtanesi Invoice'ların başındaki C ifadesi. Bunlar iadeleri ifade etmektedir. İadeler - ile işlendiği için - değerler gelmiştir

# NA değerler var, veri seti zengin olduğu için NA'leri kaldırıp kullanacağız
# Invoice'de C olanlar, quanity ve price'de eksi olanlar çıkarılacak
# outlier değerler kırpılacak

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    return dataframe

df = retail_data_prep(df)

# box-plot yönteminde aykırı değer hesaplarken iqr'da birinci çeyrek için %25, 3. çeyrek için %75 kullanılır
# burada %1 ve %99 seçilmesinin sebebi çok aykırı aşırı uçuk değerleri baskılanmak istenmesidir
# örneğin 3. çeyrek değer 95 ise bir kişi 120 yaşında ise bu değer 95'e çekilir, eşik değere baskılanır

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~(dataframe["Invoice"]).astype(str).str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe

df = retail_data_prep(df)
df.isnull().sum()
df.describe().T
#describe'a bakıldığından 0'dan küçük olma problemi kalmadı, dağılımlara bakıldığında herhangibir aykırı değer kalmadığı görülüyor


""" 
df.dtypes
df["Invoice"] = df["Invoice"].astype(str)
"""

############################################
# 2. ARL Veri Yapısını Hazırlama (Invoice-Product Matrix)
############################################

df.head()
#varmak istenen şekil aşağıdaki gibidir
#satırlarda invoice'lar sutunlarda product'lar olmalı, faturada bir ürünün olup olmaması 0-1 ile ifade edilsin
#satırlarda sepet/transaction yer alıyor
#veri seti belirli bir ülkeye indirgenerek ilerlenecek

# Description   NINE DRAWER OFFICE TIDY   SET 2 TEA TOWELS I LOVE LONDON    SPACEBOY BABY GIFT SET
# Invoice
# 536370                              0                                 1                       0
# 536852                              1                                 0                       1
# 536974                              0                                 0                       0
# 537065                              1                                 0                       0
# 537463                              0                                 0                       1


df_fr = df[df['Country'] == "France"]
# diyelimki bu online şirket avrupa'da almanya pazarına giriş yapacak, almanya pazarından gelen müşterilere ürün önermek istiyor, daha öncede hiç müşterisi yoktu bunu nasıl yapacak ?
# almanya ile benzer davranışlar sergilemesini beklediğim, elimde verileri olan  bir ülke üzerinden gidebilirim


# fatura,ürün ve hangi üründen kaçar tane alınmış
df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).head(20)
# grupby'a alma işleminden sonra pivot yapılıp, description'lar sutuna geçirilecek, bu unstack ve pivot ile gerçekleştirilebilir
# unstack ile pivot etmek buradaki isimlendirmeler değişken isimlendirmelerine çevrilecek

df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().iloc[0:5, 0:5]


#alınan ürünlere 1 alınmayanlara 0 yazmasını istiyorduk
#unstackten sonra boş olan yerler fillna() methodu ile 0 ile dolduruldu
df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0).iloc[0:5, 0:5]

# apply verilen satır ya da sutun bilgisine göre o değişkene ait ilgili kısımda işlem yapar
# applaymap ise bütün gözlemlerde gezip işlem yapar

df_fr.groupby(['Invoice', 'StockCode']). \
    agg({"Quantity": "sum"}). \
    unstack(). \
    fillna(0). \
    applymap(lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]


#ister stockCode'a göre ister Description'a göre işlem yapsın
def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)

fr_inv_pro_df = create_invoice_product_df(df_fr)

fr_inv_pro_df = create_invoice_product_df(df_fr, id=True)


def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)


check_id(df_fr, 10120)

############################################
# 3. Birliktelik Kurallarının Çıkarılması
############################################
#min_support değeri 0.01 belirlendi, eğer df'de ki sutun isimlerini kullanmak istiyorsan colnames=True yap


frequent_itemsets = apriori(fr_inv_pro_df,
                            min_support=0.01,
                            use_colnames=True)

frequent_itemsets.sort_values("support", ascending=False)
#bu liste her bir ürünün birlikte görülme olasılığı verir

rules = association_rules(frequent_itemsets,
                          metric="support",
                          min_threshold=0.01)

rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) & (rules["lift"]>5)]
# antecedents: ilk ürün, consequents: ikinci ürün, antecedent support: ilk ürünün gözlenme olasılığı, consequent support:ikinci ürünün gözlenme olasılığı, support: verilen iki ürünün birlikte gözükme olasılığıdır
# confidence : x ürünü alındığından y'nin alınması olasılığı lift : x ürünü satın alındığında y ürününün satın alınması olasılığı 17 kat artar
# leverage :kaldıraç etkisi demek, lifte benzer bir değerdir fakat leverage değeri support'u yüksek olan değerlere öncelik verme eğilimindedir bundan dolayı ufak bir yanlılığı vardır
# lift değeri ise daha az sıklıkta olmasına rağmen bazı ilişkileri yakalayabilmektedir, dolayısıyla yansız ve bizim için daha değerli bir metriktir
# conviction : y ürünü olmadan x ürününün beklenen frekansıdır, ya da x ürünü olmadan y ürününün beklenen frekansıdır

# sıralamalar istenirse lift, support değerine göre yapılabilir
# pratikte genelde şu şekilde sıralama yapılır support değeri şu değerin üzerinde olsun, confidence değeri şu değerin üzerinde olsun ...gibi

check_id(df_fr, 21086)

rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) & (rules["lift"]>5)]. \
sort_values("confidence", ascending=False)

# bir kullanıcı sepetine (21080, 21094)  iki ürünü eklediyse bu kişiye (21086) önerilecek
# çünkü iki ürün birlikte alındığında üçüncü ürünün alınma olasılığı  confidence = 0.975000 imiş
# o zaman bu kişi bunu alır,dolayısıyla bu ürün önerilmelidir


############################################
# 4. Çalışmanın Scriptini Hazırlama
############################################

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe


def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)


def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)


def create_rules(dataframe, id=True, country="France"):
    dataframe = dataframe[dataframe['Country'] == country]
    dataframe = create_invoice_product_df(dataframe, id)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules

df = df_.copy()

df = retail_data_prep(df)
rules = create_rules(df)

rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) & (rules["lift"]>5)]. \
sort_values("confidence", ascending=False)

############################################
# 5. Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak
############################################

# Örnek:
# Kullanıcı örnek ürün id: 22492
# kullanıcı sepetine bu ürünü ekledi
"""
daha önceden gerçekleşebilecek olası senaryolara karşı kime ya da hangi ürüne neyi önerebileceğimizi bir tabloda tutarız
kullanıcı login olup bir ürünü sepetine eklediği an hazır olan bu bilgiler veri tabanlarından döndürülür
"""


product_id = 22492
check_id(df, product_id)

sorted_rules = rules.sort_values("lift", ascending=False)
# buradaki sıralama confidence, support  ya da lift'e göre yapılabilir
# burası yoruma kalmış

# antecedents değişkeninde gezilecek ve consequents değişkeninde bu id'li ürünü ilk bulduğum yerdeki
# (öncesinde lift'a göre sıralanmış olduğu için) indexe karşılık ikinci ürünün ne olduğu bilgisi çekilecek


recommendation_list = []

for i, product in enumerate(sorted_rules["antecedents"]):
    for j in list(product):
        if j == product_id:
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

recommendation_list[0:3]

#bu döngüyü anlayamadım ben ???

# burada ilk öneri recommendation_liste atandı ama onun altında da ürünle ilgili eşlemeler yer alabilir

check_id(df, 22326)

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]


arl_recommender(rules, 22492, 1)
arl_recommender(rules, 22492, 2)
arl_recommender(rules, 22492, 3)





