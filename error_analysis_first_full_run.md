# Error analysis (out-of-fold predictions, full training set)

* true pairs: 7,638,365
* missed by blocking (never a candidate): 267,519 (3.50%)
* missed by the matcher (candidate rejected): 241,662 (3.16%)
* wrong merges (false positive pairs): 56,233 of 7,185,417 predicted (0.78%); on singleton S1s: 4,291

| slice | S1 entities | macro F0.5 |
|---|---|---|
| India | 883,188 | 0.9674 |
| US | 1,323,633 | 0.9732 |
| singletons | 123,247 | 0.9673 |
| 1 true match | 119,157 | 0.9224 |
| 2-4 true matches | 1,390,168 | 0.9722 |
| 5+ true matches | 574,249 | 0.9786 |

## Candidate traits: share among all true candidate pairs vs among errors

| trait | all true pairs | matcher misses (FN) | wrong merges (FP) |
|---|---|---|---|
| empty_address | 4.0% | 56.6% | 45.5% |
| indic_name | 6.8% | 1.8% | 2.4% |
| domain_name | 4.1% | 0.6% | 0.5% |
| fka_alias | 2.0% | 0.0% | 0.0% |
| name_sim<60 | 7.8% | 7.3% | 8.8% |
| first_code_differs | 20.2% | 34.7% | 36.5% |

## False positives on singleton S1s (each costs a full 1.0)

* p=0.734 | n_tset=100.00, n_jw=100.00, a_tset=86.08, num_jacc=0.33, code_first_eq=0.00, blk_frac_qmax=1.00, r_ncand=7.00
    * `S1-133681550` Norina Buchmann Allied Specialists | 2068 Northampton Street, Unit 3, Holyoke, MA | US
    * `S3-26980200` norina buchmann allied specialists corp | Unit 3, Hoyloke CITY, 2070 Northampton Street, Massachusetts | US
* p=0.788 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.48, r_ncand=38.00
    * `S1-601819809` Marrilee Tunnell Shanghai | 5709 Holy Neck Road, Suffolk City, VA | US
    * `S3-230382252` Marrilee Marrilee Tunnell Shanghai PC |  | US
* p=0.877 | n_tset=100.00, n_jw=100.00, a_tset=95.65, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=1.00, r_ncand=10.00
    * `S1-352067589` Nexix Corporation | 704 Nash Street, Rockwall, TX | US
    * `S2-309858854` Nexix Corporation Corp | 709 Nash Street, ROCKWALL, TX | US
* p=0.982 | n_tset=21.62, n_jw=43.12, a_tset=100.00, num_jacc=1.00, code_first_eq=1.00, blk_frac_qmax=0.33, r_ncand=2.00
    * `S1-925859038` Express Wireless Innovations | 204 Wayside Court, Bedford, TX | US
    * `S3-362365885` Quoaviquo | 204 Wayside Court, Bedford, Texas | US
* p=0.717 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.29, r_ncand=3.00
    * `S1-631333260` Akshay & Co Co | Mumbai, Maharashtra, Floor 2Nd, Rangoonwala Compund, Shed No 11, Mumbai | India
    * `S3-476988798` Akshay & Co. |  | India
* p=0.799 | n_tset=66.67, n_jw=86.00, a_tset=96.30, num_jacc=0.67, code_first_eq=0.00, blk_frac_qmax=1.00, r_ncand=3.00
    * `S1-106041330` Lakshmi Infrastructure Private Limited | W/O Sh Tej Pal Sharma, E-103-A, F/F, Khno-162/1, Gali No. 3, Ph-1, Bhagirathi, Vihar Near, Delhi, North East, Delhi | India
    * `S3-977762135` लक्ष्मी इंफ्रास्ट्रक्चर स्टोर्स प्राइवेट लिमिटेड | Door No 89/6 W/o Sh Tej Pal Sharma, E-103-a, F/f, Khno-162/1, Gali No. 3, Ph-1, Bhagirathi, Vihar Near, Delhi, North East, DL | India
* p=0.749 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.57, r_ncand=55.00
    * `S1-170324291` Gipson Family Practice | 1377 Bedford Avenue, Brooklyn, NY | US
    * `S2-37657162` Gipson  Family Practice PLLC |  | US
* p=0.701 | n_tset=100.00, n_jw=97.78, a_tset=85.71, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=0.54, r_ncand=4.00
    * `S1-354299557` Behavioral Health Clinic LLC | VA, Arlington County, 28 Oakland Street | US
    * `S3-94406116` Behavioral  Health Clinic (ID: 46110) | 912 Ode Street, Arlington County, Virginia | US
* p=0.868 | n_tset=100.00, n_jw=100.00, a_tset=91.80, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=1.00, r_ncand=4.00
    * `S1-74620067` Williams and Fisher Inc. | 3507 Bally Brook Drive, Unit 1B, Greensboro, NC | US
    * `S2-234440457` Williams And Fisher Inc | 3512 BALLY BROOK DRIVE, GREENSBORO, NC | US
* p=0.859 | n_tset=100.00, n_jw=100.00, a_tset=90.48, num_jacc=0.33, code_first_eq=0.00, blk_frac_qmax=1.00, r_ncand=3.00
    * `S1-232024498` All Premier Diana Inc. | 1124 25th Street, Lincoln, NE | US
    * `S2-466111398` All Premier Diana Inc. | 1135 25TH STREET, LINCOLN, NE | US
* p=0.923 | n_tset=80.00, n_jw=93.33, a_tset=100.00, num_jacc=0.67, code_first_eq=0.00, blk_frac_qmax=0.74, r_ncand=5.00
    * `S1-974034568` Pushkar & Brothers Private Limited | First Floor, Shop No 101, Shanti Nagar, Nh 8, Gurgaon, Sadar Bazar, Sadar Bazar, Gurgaon, Haryana | India
    * `S2-577561158` Pushkar & Brothehgrs Group Private Limited | NO 092 FIRST FLOOR, SHOP NO 101, SHANTI NAGAR, NH 8, GURGAON, SADAR BAZAR, SADAR BAZAR, Haryana | India
* p=0.835 | n_tset=100.00, n_jw=100.00, a_tset=100.00, num_jacc=0.50, code_first_eq=0.00, blk_frac_qmax=1.00, r_ncand=12.00
    * `S1-438616534` Rohtak Apparels Pvt. Ltd. | C/O Ajay Kaushik Dlcsupva, Campus Sector 6, Rohtak, Haryana | India
    * `S3-72412852` Rohtak Apparels Limited | No 97 C/o Ajay Kaushik Dlcsupva, Campus Sector 6, Rohtak, HR | India
* p=0.873 | n_tset=96.97, n_jw=98.82, a_tset=86.00, num_jacc=0.75, code_first_eq=0.00, blk_frac_qmax=1.00, r_ncand=1.00
    * `S1-698097371` Aradhya Infratech Pvt. Ltd. | 1 58 7 1St Floor Smr Chambers Mumbai Highway Madinaguda, Hyderabad, Telangana | India
    * `S3-141215993` Aradhya Infatech Limited | 10 58 7 1St Floor Smr Chambers Mumbai Higrway Madinaguda, Rangareddy, TG | India
* p=0.879 | n_tset=50.00, n_jw=69.96, a_tset=100.00, num_jacc=1.00, code_first_eq=1.00, blk_frac_qmax=0.62, r_ncand=4.00
    * `S1-514797095` My Technology | S.No. 741 Raimoha Raimoha, Beed, Bid, Maharashtra | India
    * `S2-665657866` माय टेक | S.NO. 741 RAIMOHA RAIMOHA, BEED, BID, महाराष्ट्र | India
* p=0.905 | n_tset=95.83, n_jw=94.30, a_tset=86.79, num_jacc=1.00, code_first_eq=1.00, blk_frac_qmax=0.70, r_ncand=8.00
    * `S1-102420172` YHR Cornerstone Cannabis Inc | 6560 Weatherfield Court, Maumee, OH | US
    * `S3-57632623` YHH C0rnerstone Cannabis Inc | 6560  Weatherfield Court, Monclova Twp, Ohio | US
* p=0.965 | n_tset=100.00, n_jw=100.00, a_tset=100.00, num_jacc=0.75, code_first_eq=1.00, blk_frac_qmax=1.00, r_ncand=20.00
    * `S1-877790227` Shanti Academy | Unit No. 696 6Th Floor, Vegas Mall Plot No 6 Sector 14, New Delhi, South West Delhi, Delhi | India
    * `S3-306302736` Shanti Academy Private Limited | Unit No. #696 7Th Floor, Vegas Mall Plot No 6 Sector 14, New Delhi, South West Delhi, दिल्ली | India
* p=0.787 | n_tset=100.00, n_jw=100.00, a_tset=100.00, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=1.00, r_ncand=5.00
    * `S1-847862438` Bulandshahr Resort Ltd | Radha Nagar, Bulandshahr, Uttar Pradesh, Bulandshahr | India
    * `S2-53015507` Pvt Bulandshahr Resort Ltd | BULANDSHAHR, Uttar Pradesh, BULANDSHAHR, DOOR NO #88 RADHA NAGAR | India
* p=0.932 | n_tset=100.00, n_jw=100.00, a_tset=82.67, num_jacc=0.50, code_first_eq=0.00, blk_frac_qmax=1.00, r_ncand=6.00
    * `S1-921253583` International Power Private Limited | Plot No. 4, Sector-4, Ballabgarh, Faridabad, Haryana | India
    * `S3-32207422` International Power Limited | Plot No. 8, Sector-4, Ballabgarh, Sahupura, HR | India
* p=0.771 | n_tset=100.00, n_jw=100.00, a_tset=91.53, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=0.79, r_ncand=9.00
    * `S1-302086733` KX Artificial | 16100 Great Oaks Drive, Unit 1603, Round Rock, TX | US
    * `S2-531589574` KX Artificial Inc | 1610 GREAT OAKS DR, ROUND ROCK, TX | US
* p=0.703 | n_tset=84.21, n_jw=94.71, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.30, r_ncand=27.00
    * `S1-109000589` Gretna E. Chiesa, MD | 7021 Knoxville Avenue, Peoria, IL | US
    * `S2-190473859` GRETNA E. CLGHIGAS, MD |  | US
* p=0.875 | n_tset=82.35, n_jw=89.29, a_tset=96.30, num_jacc=0.67, code_first_eq=0.00, blk_frac_qmax=1.00, r_ncand=8.00
    * `S1-140435819` Energy Energia Steels Company | Maharashtra, Mumbai, Mumbai City, Gala-21A, 1St Floor, Ghanshyam Ind. Estate Off. Veera Desai Road, Andheri (West), Plot-1 | India
    * `S2-648492471` Energy Energia India Co | PLOT-14, GALA-21A, 1ST FLOOR, GHANSHYAM IND. ESTATE OFF. VEERA DESAI ROAD, ANDHERI (WEST), MUMBAI, Maharashtra | India
* p=0.762 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.43, r_ncand=75.00
    * `S1-905686151` Gertie's Colonial Holdings | 1366 Western Avenue, Unit Apartment 2, Saint Paul, MN | US
    * `S2-967394858` Gertie's Colonial Holdings (LLC) |  | US
* p=0.791 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.46, r_ncand=69.00
    * `S1-981055494` Tirumala & Sons Private Limited | #127, First Floor, 12Th Cross, Domlur, Bangalore, Karnataka | India
    * `S3-402728050` Tirumala & Sons Pvt. |  | India
* p=0.821 | n_tset=100.00, n_jw=100.00, a_tset=89.36, num_jacc=0.50, code_first_eq=0.00, blk_frac_qmax=1.00, r_ncand=4.00
    * `S1-525490934` Pakhi & Sons Private Limited | 67/3/3, Sk Compound, Indore, Madhya Pradesh | India
    * `S3-546863528` M/s Pakhi & Sons Limited | No. 607 67/3/5, Sk Compound, Indore, MP | India
* p=0.750 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.47, r_ncand=34.00
    * `S1-885721096` Family Federal Physicians LLC | 2562 Country Estates Road, Milo, NY | US
    * `S3-380918080` The Family Féderal Physicians LLC |  | US

## False positives on non-singleton S1s

* p=0.932 | n_tset=100.00, n_jw=100.00, a_tset=96.97, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=0.56, r_ncand=2.00
    * `S1-400822065` Pediatric Health Inc. | MN, 3000 Independence Avenue, New Hope | US
    * `S3-646892102` PEDIATRIC HEALTH INC | 3001- Independence Avenue, New Hope, Minnesota | US
* p=0.743 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.54, r_ncand=23.00
    * `S1-525677632` King, Violet, DO PLLC | 120 Orchard Drive, Purcellville Town, VA | US
    * `S3-220309343` King, Víolet, DO |  | US
* p=0.959 | n_tset=100.00, n_jw=94.55, a_tset=100.00, num_jacc=0.50, code_first_eq=0.00, blk_frac_qmax=0.91, r_ncand=6.00
    * `S1-424776316` Tessa Jaso Diamond Grade LLC | 109 Hopkins Street, Unit 110, Wakefield, MA | US
    * `S3-65454598` tessa jaso diamond grade partners | 110 Hopkins Street, # 110, Wakefield, Massachusetts | US
* p=0.788 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.36, r_ncand=22.00
    * `S1-780235199` Strategic Sterling Illinois PLLC | 284 Courtney Drive, Walhalla, SC | US
    * `S2-154392091` Strategic Ster1ing Illinois |  | US
* p=0.853 | n_tset=100.00, n_jw=100.00, a_tset=93.02, num_jacc=0.75, code_first_eq=0.00, blk_frac_qmax=0.89, r_ncand=10.00
    * `S1-101259452` Engineering Bhairavkripa (india) Pvt Ltd | Apartment No. Nl - 1A/29/01, First Floor Sector - 10, Nerul, Navi Mumbai, Thane, Maharashtra | India
    * `S3-317239459` Engineering Bhairavkripa (india) Limited | Apartment No. Nl - 1A/29/3, First Floor Sector - 10, Nerul, Mumbai, Thane, MH | India
* p=0.765 | n_tset=85.11, n_jw=91.95, a_tset=91.67, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=0.28, r_ncand=1.00
    * `S1-690382162` Best Insurance Holdings | 1607 Acadia Street, Durham, NC | US
    * `S3-988881910` Best Igsaharnce Holdings Holdings | ##1611 Acadia Street, Durham, North Carolina | US
* p=0.827 | n_tset=100.00, n_jw=87.83, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.14, r_ncand=12.00
    * `S1-239091207` Dorf Blue Environmental LLC | 1211 Gold Star Highway, Groton, CT | US
    * `S3-718195795` Dorf Blue |  | US
* p=0.783 | n_tset=100.00, n_jw=100.00, a_tset=72.22, num_jacc=0.33, code_first_eq=0.00, blk_frac_qmax=0.80, r_ncand=3.00
    * `S1-117479529` Physical Therapy Group | IL, Unit 40, 106 Jefferson Street, Shorewood | US
    * `S3-39741093` physical therapy group ltd | Illinois, Incorporated, 108 Jefferson St, # 40 | US
* p=0.722 | n_tset=100.00, n_jw=100.00, a_tset=89.29, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=0.61, r_ncand=3.00
    * `S1-875973578` Legacy Coastal Inc. | 21243 Briggs Road, OH, Spencerville | US
    * `S2-335043537` Legacy Cóastal Inc | 21256 1/2 BRIGGS RD, SPENCERVILLE, OH | US
* p=0.751 | n_tset=57.78, n_jw=80.96, a_tset=93.33, num_jacc=0.50, code_first_eq=0.00, blk_frac_qmax=0.42, r_ncand=4.00
    * `S1-5891803` Super Indo Foods Pvt Ltd | Gat No. 50/3, Jamgaon Road, Barshi, Solapur, Maharashtra | India
    * `S2-721236646` सुपर इंडो फूड्स होटल प्रा. लि. | GAT NO. 50, BARSHI, SOLAPUR, Maharashtra | India
* p=0.769 | n_tset=100.00, n_jw=100.00, a_tset=100.00, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=1.00, r_ncand=14.00
    * `S1-699189055` Lotus Institute of Technology | First & Second Floor, Udb Hallmark Complex Gautam Marg, Vaishali Nagar, Jaipur, Rajasthan | India
    * `S3-341186119` Lotus Institute OF Technology (LLP) | No 84 First & Second Floor, Udb Hallmark Complex Gautam Marg, Vaishali Nagar, Jaipur, RJ | India
* p=0.968 | n_tset=100.00, n_jw=100.00, a_tset=100.00, num_jacc=1.00, code_first_eq=1.00, blk_frac_qmax=1.00, r_ncand=5.00
    * `S1-680069674` Rocky Metro Consulting PLLC | 7 San Antonio Drive, Cedar Crest, NM | US
    * `S3-413664830` Inc Rocky Metro Consulting | 7 San Antonio Drive, Cedar Crest, New Mexico | US
* p=0.859 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.38, r_ncand=12.00
    * `S1-25141042` Moran's Precision Advisory | 90 Pleasant Street, Unit R, Medfield, MA | US
    * `S2-844851620` Moran's Precision (Advisory) |  | US
* p=0.889 | n_tset=70.00, n_jw=78.65, a_tset=98.77, num_jacc=0.67, code_first_eq=0.00, blk_frac_qmax=0.72, r_ncand=3.00
    * `S1-263348186` Green Red Trading Limited | 17/A, Santosha Park, 068/65/0066/0001/R, Nr. Jayantilal Park, Opp.Ashok Vatika, Bopal, Ahmedabad, Gujarat | India
    * `S2-143825374` ગ્રીન રેડ ટ્રેડિંગ પ્રા. લિ. | 0/A, SANTOSHA PARK, 068/65/0066/0001/R, NR. JAYANTILAL PARK, OPP.ASHOK VATIKA, BOPAL, AHMEDABAD, BOPAL, Gujarat | India
* p=0.802 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.35, r_ncand=109.00
    * `S1-590457170` Crowder Piedmont Keycorp LLC | 1322 Bee Rock Road, Monterey, TN | US
    * `S3-112590227` Crowder Piedmont Keycorp of |  | US
* p=0.762 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.47, r_ncand=13.00
    * `S1-459014720` Adami Biosciences Corp | 320 Pine Street, Richmond City, VA | US
    * `S3-650367366` Adami Biosciences Biosciences |  | US
* p=0.715 | n_tset=100.00, n_jw=100.00, a_tset=94.87, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=0.55, r_ncand=3.00
    * `S1-410090999` Puttaparthi Hardware Private Limited | 407- Dharma Block, Gokul Sai Magan Apartment Gokul Sai Magan Apartment, Puttaparthi, Anantapur, Andhra Pradesh | India
    * `S3-587444022` Puttaparthi Hardware Limited Limited | 409- Dharma Block, Puttaparthi, Anantapur, AP | India
* p=0.817 | n_tset=100.00, n_jw=94.81, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.07, r_ncand=55.00
    * `S1-447438148` Innovative Materials Brands Inc | 6501 Donnegal Way, Sykesville, MD | US
    * `S3-302739365` Innovative Brands Materials Inc. |  | US
* p=0.897 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.41, r_ncand=5.00
    * `S1-88852221` Bevington & Gryder LLC | 234 Elm Street, West Bridgewater, MA | US
    * `S2-180025494` Bevington & Grydér |  | US
* p=0.762 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.31, r_ncand=50.00
    * `S1-521089539` Niia Enterprises LLC | IN, 2902 Overlook Drive, Fort Wayne | US
    * `S3-394390506` Niia Enterprises |  | US
* p=0.918 | n_tset=100.00, n_jw=100.00, a_tset=89.47, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=1.00, r_ncand=3.00
    * `S1-993443704` Beckett, Smith and Orosco, Inc. | 108 Main Street, Almyra, AR | US
    * `S2-660206478` Beckett, Smith Inc Orosco, and | 117 MAIN STREET, PMB 6299, ALMYRA, AR | US
* p=0.835 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.31, r_ncand=4.00
    * `S1-187260103` Duncan Lincoln Inc | 611 64 Birch Street, New Berlin, IL | US
    * `S2-322552262` Duncan Lincoln |  | US
* p=0.766 | n_tset=100.00, n_jw=100.00, a_tset=96.67, num_jacc=0.67, code_first_eq=0.00, blk_frac_qmax=0.66, r_ncand=5.00
    * `S1-621195938` Ram Producer Limited | 66/1/1 Strand Bank Road, Kolkata, Calcutta, West Bengal | India
    * `S2-377007449` Ram Producer Private Limited | West Bengal, 66/1/6 STRAND BANK ROAD, KOLKATA | India
* p=0.916 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.13, r_ncand=63.00
    * `S1-390462083` Siddhanath Mauli Private Limited | Spire T110, Survey No. 83/1, Hyderabad Knowledge City, Raidurga Village, Serilingampally, Hyderabad, 3Rd Floor, Shaikpet, Telangana | India
    * `S2-77862033` Siddhanath Mauli Private Ltd |  | India
* p=0.823 | n_tset=100.00, n_jw=92.63, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.14, r_ncand=59.00
    * `S1-908107751` Rachita & Sons Private Limited | Tower-C2/103, Sec110A/111, Puri Diplomatic Greens, Palam Vihar, Gurgaon, Haryana | India
    * `S2-892497406` Rachita Sons Pvt Ltd Center |  | India

## Matcher false negatives (true candidate rejected)

* p=0.073 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.25, r_ncand=22.00
    * `S1-357309554` Lower Animal Hospital | 3172 Montego Lane, Bldg 4, Maineville, OH | US
    * `S3-620044785` Lower Animal-Hospital |  | US
* p=0.290 | n_tset=100.00, n_jw=100.00, a_tset=89.29, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=0.57, r_ncand=8.00
    * `S1-377283955` Department of Sanitation | 12425 Brightwater Lane, Henrico County, VA | US
    * `S2-289861567` Department of Sanitation Corporation | 12427 BRIGHTWATER LN, HENRICO, VA | US
* p=0.636 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.17, r_ncand=39.00
    * `S1-522012876` Total Wealth Networks Inc | 4402 Riverchase Drive, Unit 4212, Phenix City, AL | US
    * `S3-834122724` Total Wealth Networks Incorporated |  | US
* p=0.639 | n_tset=69.23, n_jw=46.26, a_tset=96.88, num_jacc=1.00, code_first_eq=1.00, blk_frac_qmax=0.66, r_ncand=1.00
    * `S1-316063895` Hunt, Elissa, P.A. | 6772 Doric Place, Prescott Valley, AZ | US
    * `S2-338755307` Efiasbg,fa Hunt, [P.A.] | 6772 DORIC PL, PRESCOTT VALELY, AZ | US
* p=0.240 | n_tset=100.00, n_jw=92.50, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.23, r_ncand=68.00
    * `S1-562745749` Stoltzfus Great Carolina LLC | 301 Garden Circle, Yorkville, IL | US
    * `S3-478233655` Stoltzfus  Great |  | US
* p=0.039 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.15, r_ncand=52.00
    * `S1-678176551` Golden Business Private Limited | Prabhalata First Floor No 8, 6Th Cross V V Mohalla, Mysore, Karnataka | India
    * `S3-982136324` Golden Búsiness Private Limited |  | India
* p=0.590 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.22, r_ncand=117.00
    * `S1-13666072` Jaipur Finsec Limited | Shop No B 32, Kubercomplex Ghandhi Path, Jaipur, Rajasthan | India
    * `S2-59355993` Jaipur Finsec Ltd |  | India
* p=0.662 | n_tset=62.50, n_jw=83.43, a_tset=97.78, num_jacc=0.33, code_first_eq=0.00, blk_frac_qmax=1.00, r_ncand=5.00
    * `S1-97677009` White Power Pvt Ltd | Unit No 302, Building No-8 Jogani Industrial Estate, Sion, Chiunabhat, I, Mumbai, Mumbai City, Maharashtra | India
    * `S2-769130615` White पावर प्रा. लि. | UNIT NO 30, BUILDING NO-8 JOGANI INDUSTRIAL ESTATE, SION, CHIUNABHAT, I, MUMBAI, Maharashtra | India
* p=0.080 | n_tset=75.00, n_jw=90.33, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.13, r_ncand=17.00
    * `S1-717344371` Ganjam (india)private Private Limited | Pratima Nivas, At/Po-Kanisi Hatta, Ganjam, Orissa | India
    * `S3-920595981` Ganjam (indai)prviate Private Limited |  | India
* p=0.020 | n_tset=100.00, n_jw=95.38, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.30, r_ncand=88.00
    * `S1-125445594` Cure Pizza | 2491 Martin Avenue, Hempstead, NY | US
    * `S2-731004263` Cure (Pizza) (ID: 89437) |  | US
* p=0.424 | n_tset=100.00, n_jw=100.00, a_tset=67.92, num_jacc=0.33, code_first_eq=0.00, blk_frac_qmax=0.36, r_ncand=6.00
    * `S1-511906908` Systems Green Power Pvt Ltd | B-31-32, Haryana, Gurgaon, Idc, Mehrauli Road, Gurgaon, Gurgaon | India
    * `S3-127048170` Systems Green Power Pvt | Gurgaon, HR, Palam, B-31-3 | India
* p=0.029 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.15, r_ncand=86.00
    * `S1-53799309` City International Private Limited | 408, 4Th Floor, Tower, Pavilion Heights 1, Jaypee, Noida, Gautam Buddha Nagar, Uttar Pradesh | India
    * `S2-776269107` City International Private Ltd |  | India
* p=0.397 | n_tset=100.00, n_jw=100.00, a_tset=91.18, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=0.43, r_ncand=5.00
    * `S1-90566860` #5 Biosciences | 214 -90 Wythe Creek Road, Poquoson City, VA | US
    * `S3-493443460` #5 Biosciences LLC | 21225 Wythe Creek Rd, Poquoson City, Virginia | US
* p=0.469 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.32, r_ncand=61.00
    * `S1-621858889` Platinum Malt | 1400 Arno Street, Albuquerque, NM | US
    * `S2-599809325` PLATINUM MALT LP |  | US
* p=0.276 | n_tset=100.00, n_jw=100.00, a_tset=98.48, num_jacc=0.60, code_first_eq=0.00, blk_frac_qmax=0.73, r_ncand=7.00
    * `S1-787045764` Roopavi Solutions Private Limited | Revenue Colony 2Nd Line Machavaram Down, Andhra Pradesh, Krishna, 32-41-30, Vijayawada | India
    * `S3-531694280` Roopavi Solutions | 82-41-30, Revenue Colony 2Nd Line Machavaram Down, Vijayawada, Krishna, AP | India
* p=0.628 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.35, r_ncand=276.00
    * `S1-221413242` Moore Spade LLC | 712 C Alley, State College, PA | US
    * `S2-763283595` Moore Spade |  | US
* p=0.039 | n_tset=100.00, n_jw=90.77, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.08, r_ncand=29.00
    * `S1-397821325` Tri-State Best | 519 Spring Meado Street, Stephenville, TX | US
    * `S3-914413881` tri-state best Enterprises |  | US
* p=0.618 | n_tset=82.76, n_jw=94.37, a_tset=93.75, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=0.18, r_ncand=7.00
    * `S1-45558592` Regional Senior Inc | 722 Mount Vernon Avenue, Chattanooga, TN | US
    * `S2-904244312` Regional  Socir Inc | 717 MT VERNON AVE, CHATTANOOGA, TN | US
* p=0.057 | n_tset=100.00, n_jw=94.78, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.14, r_ncand=29.00
    * `S1-756450337` Unified Nutrition Group | 3606 Mustang Lane, Manvel, TX | US
    * `S2-909650338` UNIFIED [NUTRITION] |  | US
* p=0.013 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.28, r_ncand=53.00
    * `S1-280310871` Economic Development Council | 8051 83, Villenova, NY | US
    * `S2-222362429` Economic Development-Council Corp |  | US
* p=0.396 | n_tset=77.78, n_jw=90.25, a_tset=96.43, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=0.58, r_ncand=4.00
    * `S1-833433425` Yuan Tbk | 1303 Stafford Road, Sherwood, AR | US
    * `S3-409854396` Ymuena Tbk | 6303 Stafford Rd, Sherwood, Arkansas | US
* p=0.534 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.42, r_ncand=38.00
    * `S1-601025738` Roify Anchor Calisa LP | MD, 36884 Park View Lane, Mechanicsville | US
    * `S2-169849130` ROIFY ANCHOR CÁLISA LP |  | US
* p=0.075 | n_tset=91.43, n_jw=91.84, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.10, r_ncand=338.00
    * `S1-812322119` New Delhi Computers LLP | G-1387 2Nd Floor Chittaranjan Park, New Delhi, South Delhi, Delhi | India
    * `S2-543367975` New Di Computers LLP |  | India
* p=0.492 | n_tset=100.00, n_jw=100.00, a_tset=96.47, num_jacc=0.33, code_first_eq=0.00, blk_frac_qmax=0.63, r_ncand=3.00
    * `S1-818905348` Mc & Associates Ltd | # 3/235, Sector 3, Vishal Khand, Gomti Nagar, Lucknow, Uttar Pradesh | India
    * `S3-489516172` Mc & Associates | 030, Sector 3, Vishal Khand, Gomti Nagar, Lucknow, UP | India
* p=0.021 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.30, r_ncand=46.00
    * `S1-535719608` Hunger Foundation | 5000 913a, Godley, TX | US
    * `S2-773798547` Hunger Foundation Corp |  | US
