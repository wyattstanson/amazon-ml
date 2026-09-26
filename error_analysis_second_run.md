# Error analysis (out-of-fold predictions, full training set)

* true pairs: 7,638,365
* missed by blocking (never a candidate): 207,576 (2.72%)
* missed by the matcher (candidate rejected): 190,968 (2.50%)
* wrong merges (false positive pairs): 24,385 of 7,264,206 predicted (0.34%); on singleton S1s: 2,919

| slice | S1 entities | macro F0.5 |
|---|---|---|
| India | 883,188 | 0.9740 |
| US | 1,323,633 | 0.9818 |
| singletons | 123,247 | 0.9803 |
| 1 true match | 119,157 | 0.9297 |
| 2-4 true matches | 1,390,168 | 0.9800 |
| 5+ true matches | 574,249 | 0.9853 |

## Candidate traits: share among all true candidate pairs vs among errors

| trait | all true pairs | matcher misses (FN) | wrong merges (FP) |
|---|---|---|---|
| empty_address | 4.0% | 64.0% | 15.9% |
| indic_name | 6.7% | 1.4% | 3.4% |
| domain_name | 4.7% | 0.5% | 1.1% |
| fka_alias | 2.0% | 0.0% | 0.0% |
| name_sim<60 | 6.3% | 8.5% | 14.0% |
| first_code_differs | 20.2% | 25.2% | 55.4% |

## False positives on singleton S1s (each costs a full 1.0)

* p=0.681 | n_tset=100.00, n_jw=100.00, a_tset=96.55, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=1.00, r_ncand=4.00
    * `S1-221371125` Signature Wholesale Services Inc. | Conroe, TX, 10718 Juan Diego Court | US
    * `S2-163047774` SIGNATURE WHOLESALE SERVICES INC. | 10721 JUAN DIEGO CT, CONROE, TX | US
* p=0.999 | n_tset=50.00, n_jw=65.38, a_tset=86.84, num_jacc=1.00, code_first_eq=0.00, blk_frac_qmax=1.00, r_ncand=4.00
    * `S1-837021918` Rise Imaging Pvt Ltd | F3-20/12/1-3 B.B.T Road, Bye Lane-3, Bishnupur - I, South 24 Parganas, West Bengal | India
    * `S3-520552730` Pvt Rise Services Ltd | F3-20/12/1-12 B.b.t Road, Bye Lane-3, South 24 Parganas, WB | India
* p=0.859 | n_tset=100.00, n_jw=100.00, a_tset=89.36, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=0.50, r_ncand=2.00
    * `S1-683374180` Tiex | 1321 Lake Drive, Unit Unit 234, Chanhassen, MN | US
    * `S2-664030016` Tiex LLC | 1323 LAKE DRIVE, CHANHASSEN, MN | US
* p=0.865 | n_tset=100.00, n_jw=100.00, a_tset=96.67, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=1.00, r_ncand=3.00
    * `S1-999181727` Foot & Ankle Specialists L.L.C. | 223 Camellia Avenue, Greenville, AL | US
    * `S2-750371122` Foot & Ankle Specialists LLC | GREENVILLE, 228 CAMELLIA AVENUE, AL | US
* p=0.949 | n_tset=100.00, n_jw=100.00, a_tset=94.29, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=1.00, r_ncand=5.00
    * `S1-268273241` Family Committee | 125 Morning Dove Bend, Sierra Vista, AZ | US
    * `S2-30250770` FAMILY COMMITTEE INC | SIERRA VISTA, 134-D MORNING DOVE BEND, AZ | US
* p=0.867 | n_tset=100.00, n_jw=100.00, a_tset=71.11, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=1.00, r_ncand=3.00
    * `S1-586749245` YB China Inc. | 67 Cross Pond Road, Lewisboro, NY | US
    * `S3-361849278` YB China Inc. | 88 Cross Pond Road, Pound Ridge, New York | US
* p=0.986 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.63, r_ncand=3.00
    * `S1-585636197` Andrades Diamond Portfolio LLC | 4417 Commercial Avenue, Portland, OR | US
    * `S3-220114614` Andrades Díamond Portfolio LLC |  | US
* p=0.984 | n_tset=100.00, n_jw=100.00, a_tset=93.10, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=1.00, r_ncand=14.00
    * `S1-300135228` Primary Care Empire Partners Inc. | 397 Big Elk Chapel Road, Elkton, MD | US
    * `S3-42698181` Primary Care Empire Partners Partners Inc | Maryland, 406 Big Elk Chapel Road, Elkton | US
* p=0.823 | n_tset=100.00, n_jw=100.00, a_tset=98.73, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=1.00, r_ncand=20.00
    * `S1-57406165` Pioneer Technology | 2003, Bankston, Rodas Enclave, Hiranandani Estate Patlipada, Ghodbundar Road, Thane, Maharashtra | India
    * `S3-179768697` Pioneer Technology Public Limited | 2007, Bankston, Rodas Enclave, Hiranandani Estate Patlipada, Ghodbundar Road, Thane, महाराष्ट्र | India
* p=0.569 | n_tset=100.00, n_jw=100.00, a_tset=93.33, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=0.61, r_ncand=3.00
    * `S1-28311865` Fresh Massage | 50 Richmond Road, Unit Unit 11, Putnam, CT | US
    * `S2-777953775` Fresh Massage Inc | 52 RICHMOND RD, PUTNAM, CT | US
* p=0.934 | n_tset=100.00, n_jw=100.00, a_tset=95.65, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=1.00, r_ncand=5.00
    * `S1-382444351` Education Initiative | Hot Springs National Park, AR, 280 Candleberry Circle | US
    * `S2-71892546` Education Initiative Inc | AR, 281 CANDLEBERRY CIRCLE, HOT SPRINGS NATIONAL PARK, PMB 8328 | US
* p=0.942 | n_tset=100.00, n_jw=100.00, a_tset=96.88, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=1.00, r_ncand=11.00
    * `S1-5200038` Fredrika Keil Tactical L.L.C. | 8822 Lew Wallace Road, Frederick, MD | US
    * `S2-660082727` FREDRIKA KEIL TACTICAL LLC | #8823 LEW WALLACE ROAD, FREDERICK, MD | US
* p=1.000 | n_tset=100.00, n_jw=100.00, a_tset=64.15, num_jacc=1.00, code_first_eq=1.00, blk_frac_qmax=1.00, r_ncand=1.00
    * `S1-786986225` Pioneer New Finance | 2-2-642/1/1, Amberpet, Hyderabad, Telangana | India
    * `S3-406032719` Pioneer New Finance Public Limited | 2-2-642/1/1, Sangareddy Ap, తెలంగాణ | India
* p=0.931 | n_tset=100.00, n_jw=91.43, a_tset=98.18, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=0.83, r_ncand=6.00
    * `S1-21858759` Bright Vance | 710 Wiley Avenue, Barnesville, OH | US
    * `S3-279891825` Bright Vance  [Partners] | 71 Wiley Avenue, Barnesville, Ohio | US
* p=0.993 | n_tset=62.50, n_jw=83.97, a_tset=100.00, num_jacc=1.00, code_first_eq=0.00, blk_frac_qmax=1.00, r_ncand=19.00
    * `S1-908617065` Black Marketing Private Limited | A-127, Ground Floor, Dda Flat, Bharthal, Village, Sector 26, Dwarka, New Delhi, South West Delhi, Delhi | India
    * `S2-443469694` B1ack Engineering Private (Limited) | SOUTH WEST DELHI, GROUND FLOOR, DDA FLAT, BHARTHAL, VILLAGE, SECTOR 26, DWARKA, A-127, NEW DELHI, Delhi | India
* p=0.659 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.30, r_ncand=62.00
    * `S1-724259215` Vaayuita Services Private Limited | Bangalore, 205 Niranjan Nectar 85 Bilekahalli Road, Karnataka | India
    * `S2-228209409` Vaayuita Ltd [Services] |  | India
* p=0.510 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.36, r_ncand=7.00
    * `S1-284891442` Piedmont For | Fargo, ND, Unit Apartment 5, 1315 8th Avenue | US
    * `S2-135615085` PIEDMONT  FOR |  | US
* p=0.664 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.55, r_ncand=2.00
    * `S1-785405444` Arroyo Pennantpark Inc. | 10856 Nord Avenue, Bloomington, MN | US
    * `S3-444305706` Arroyo Pennantpark |  | US
* p=0.799 | n_tset=59.46, n_jw=83.00, a_tset=83.33, num_jacc=1.00, code_first_eq=1.00, blk_frac_qmax=0.18, r_ncand=1.00
    * `S1-785785229` Surya Solutions Private Limited | Empaire Square, Nr. Empire Tharti - 6, Mahendranagar Chokdi, Morbi, Rajkot, Gujarat | India
    * `S2-769526026` સૂર્ય સોલ્યુશન્સ બેકરી પ્રાઇવેટ લિમિટેડ | NO 006 EMPAORE SQUARE, MORBI, RAJKOT, ગુજરાત | India
* p=1.000 | n_tset=100.00, n_jw=100.00, a_tset=100.00, num_jacc=1.00, code_first_eq=1.00, blk_frac_qmax=1.00, r_ncand=4.00
    * `S1-500427837` Vivia Thomas, DO | 1700 Lee Road, Cleveland Heights, OH | US
    * `S3-775074425` Vivia Thomas, DO L.L.C. | 1700 Lee Rd, Cleveland, Ohio | US
* p=0.813 | n_tset=100.00, n_jw=100.00, a_tset=90.91, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=1.00, r_ncand=1.00
    * `S1-189764908` Premier Environmental Corporation | 7921 Ramel Road, Town Of Solon Springs, WI | US
    * `S3-998285367` Premier Environmental Córporation Corp | 7925- Ramel Rd, Solon Springs, Wisconsin | US
* p=0.990 | n_tset=100.00, n_jw=94.78, a_tset=85.71, num_jacc=1.00, code_first_eq=1.00, blk_frac_qmax=1.00, r_ncand=2.00
    * `S1-980158348` Family Associates Co | 72 Geneva Drive, Kent, NY | US
    * `S2-373787553` Family Associates Group | 72 GENEVA DR, CARME LCITY, NY | US
* p=1.000 | n_tset=100.00, n_jw=93.64, a_tset=70.59, num_jacc=0.50, code_first_eq=0.00, blk_frac_qmax=0.41, r_ncand=10.00
    * `S1-80407064` Ear Nose & Throat Center | 1228 Royal Bend Drive, Fl 0, Eagle Pass, TX | US
    * `S3-756801455` Ear Nose & Throat | Wagon Wheel Rd, Floor 0, Eagle Pass, Texas | US
* p=0.541 | n_tset=11.76, n_jw=41.43, a_tset=100.00, num_jacc=1.00, code_first_eq=1.00, blk_frac_qmax=0.71, r_ncand=4.00
    * `S1-615455414` Juniper | 29 Hickory Street, Lester Prairie, MN | US
    * `S3-422545063` Deltacalo | 29 Hickory St, Lester Prairie, Minnesota | US
* p=0.743 | n_tset=100.00, n_jw=100.00, a_tset=96.63, num_jacc=0.50, code_first_eq=0.00, blk_frac_qmax=1.00, r_ncand=5.00
    * `S1-117751041` Akr Services Pvt Ltd | 4, Kamal Kunj, Road No. 4, Jvpd Scheme, Vile Parle (West), Mumbai, Maharashtra | India
    * `S2-997421001` Akr-5ervices Limited | 6/4, KAMAL KUNJ, ROAD NO. 4, JVPD SCHEME, VILE PARLE (WEST), Maharashtra | India

## False positives on non-singleton S1s

* p=0.968 | n_tset=100.00, n_jw=91.00, a_tset=100.00, num_jacc=1.00, code_first_eq=1.00, blk_frac_qmax=0.91, r_ncand=5.00
    * `S1-219279567` Cross and Jones LLC | 1411 Calle Pensamiento, Thousand Oaks, CA | US
    * `S3-150458259` Cross and Jones Partners | 1411 Calle Pensamiento, Thousand Oaks, California | US
* p=0.915 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.29, r_ncand=24.00
    * `S1-470291730` Hudson Inc | 5203 Beachside Drive, Minnetonka, MN | US
    * `S3-925914335` Hudson Inc |  | US
* p=0.989 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.32, r_ncand=1.00
    * `S1-191616114` Big Grill Center | 215 Elm Street, Fl 0, Alvord, TX | US
    * `S3-849778468` Big Grill [Center] |  | US
* p=0.777 | n_tset=100.00, n_jw=100.00, a_tset=60.00, num_jacc=0.25, code_first_eq=0.00, blk_frac_qmax=1.00, r_ncand=2.00
    * `S1-655148812` Vision Group | 12639 47th Drive, Phoenix, AZ | US
    * `S2-713917041` Vision Group  Group | 12641-12645 47TH DR, GLENDALE, AZ | US
* p=0.997 | n_tset=100.00, n_jw=97.37, a_tset=100.00, num_jacc=1.00, code_first_eq=1.00, blk_frac_qmax=0.92, r_ncand=7.00
    * `S1-537627387` Burns, Reese & Shell Pharmaceuticals | 1371 Tower Road, Midlothian, TX | US
    * `S3-294626575` BURNS, REESE & SHELL PHARMACEUTICALS WEST | 1371 Tower Rd, Midlothian, Texas | US
* p=1.000 | n_tset=100.00, n_jw=100.00, a_tset=100.00, num_jacc=1.00, code_first_eq=1.00, blk_frac_qmax=1.00, r_ncand=3.00
    * `S1-748153340` Ear Nose & Throat Health | 567 Sing Sing Road, Big Flats, NY | US
    * `S3-960637568` Ear Nose & Throat Health Corp | 567 Sing Sing Rd, Big Flats, New York | US
* p=0.890 | n_tset=100.00, n_jw=100.00, a_tset=96.88, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=1.00, r_ncand=7.00
    * `S1-801056273` E J Scott Imaging Inc. | 1201 Ewing Creek Drive, Nashville, TN | US
    * `S3-615545545` E J  Scott Imaging Inc | Nashville, Tennessee, 1214 Ewing Creek Dr | US
* p=0.853 | n_tset=100.00, n_jw=100.00, a_tset=91.23, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=0.56, r_ncand=60.00
    * `S1-549812500` Cozy Cafe Corporation | 2926 Ramblin Road, Evansville, IN | US
    * `S3-648147293` Cozy Cafe Corp | 2935. Ramblin Rd, Evanville, Indiana | US
* p=0.859 | n_tset=100.00, n_jw=100.00, a_tset=96.43, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=0.78, r_ncand=7.00
    * `S1-511342255` Foot & Ankle Coastal Group | NC, Rocky Mount, 305 Old Coach Road | US
    * `S3-757857481` Foot & Ankle Coastal Group Group | North Carolina, Rocky Mount, 307 Old Coach Rd | US
* p=0.727 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=1.00, r_ncand=100.00
    * `S1-375312167` Lavinie Arroyo Green Maywood LLC | 2787 28, Milford, NY | US
    * `S2-8198075` Lavinie Arroyo-Green Maywood LLC |  | US
* p=0.829 | n_tset=70.59, n_jw=51.37, a_tset=90.48, num_jacc=0.33, code_first_eq=0.00, blk_frac_qmax=0.27, r_ncand=1.00
    * `S1-116365686` Empire Fund | 100 7th Avenue, City Of Park Falls, WI | US
    * `S2-825770187` SERVICES EMPIRE GROUP | 102 7TH AVE, N/A, PARK FALLS, WI | US
* p=0.940 | n_tset=100.00, n_jw=100.00, a_tset=75.00, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=0.61, r_ncand=2.00
    * `S1-581841094` Rapid Ridge Inc. | 1201 Half Street, Unit 227, Washington, DC | US
    * `S2-459097623` Rapid Ridge Inc | 1203 HALF STRET, WASHINGTON, DC | US
* p=0.884 | n_tset=100.00, n_jw=100.00, a_tset=98.31, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=0.84, r_ncand=6.00
    * `S1-466655244` Pediatric Dentistry Clinic | 1273 Sheffield Boulevard, Houston, TX | US
    * `S2-800083485` Pediatric Dentistry Clinic L.L.C. L.L.C. | 127 SHEFFIELD BLVD, HOUSTON, TX | US
* p=0.708 | n_tset=100.00, n_jw=100.00, a_tset=83.33, num_jacc=0.13, code_first_eq=0.00, blk_frac_qmax=0.80, r_ncand=3.00
    * `S1-19598011` Alpha New Infotech Private Limited | Kh. No. 28/22/2, 23/2, 38/2, 3Min, 8Min, 9/10/11/12/1, Ground Floor 19/1, 20 Village, Akbarpur Majra, Delhi, North West Delhi, Delhi | India
    * `S3-768182180` Alpha New Infotech Limited | Kh. No. 28/22/4, North West Delhi, DL | India
* p=0.802 | n_tset=100.00, n_jw=100.00, a_tset=98.59, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=0.67, r_ncand=7.00
    * `S1-168522732` Henry, Bush and Hotz | 3031 Grand Way Avenue, St. George City, LA | US
    * `S2-465356543` Henry, Bush and Hotz Corp | 303 GRAND WAY AVENUE, ST. GEORGE CITY, LA | US
* p=0.982 | n_tset=100.00, n_jw=100.00, a_tset=97.06, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=0.82, r_ncand=5.00
    * `S1-577543438` Norridgewock Veterans Select Project | ME, Norridgewock, 211 Wilder Hill Road | US
    * `S2-404083714` Norridgewock Veterans Select Project Inc | 213 WILDER HILL ROAD, NORRIDGEWOCK, ME | US
* p=0.962 | n_tset=37.21, n_jw=57.71, a_tset=100.00, num_jacc=1.00, code_first_eq=1.00, blk_frac_qmax=0.29, r_ncand=2.00
    * `S1-846513517` RT Star Fortunex | 626 Webb Road, Wadesboro, NC | US
    * `S2-872306016` Chiropractic Modem Partners Corp | 626 WEBB ROAD, WADESBORO, NC | US
* p=0.884 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.23, r_ncand=2.00
    * `S1-137698382` Back Alley Pizza Center | 229 Summer Rest Road, Wilmington, NC | US
    * `S2-552482289` Back Alley  Pizza Center |  | US
* p=0.887 | n_tset=100.00, n_jw=100.00, a_tset=93.53, num_jacc=1.00, code_first_eq=0.00, blk_frac_qmax=0.70, r_ncand=8.00
    * `S1-464654884` Jana Sangh | D.No: 16/11/741/B/2To13/2, F No 202, Vinayaka Towers Beside Avanthi Degree College, Moosaramba, Gh Road, Hyderbad, Hyderabad, Telangana | India
    * `S2-257333162` Jana Sangh Pvt Ltd | D.NO: 16/11/741/B/2TO13/11, F NO 202, VINAYAKA TOWERS BESIDE AVANTHI DEGREE COLLEGE, MOOSARAMBA, GH ROAD, HYDERBAD, Telangana | India
* p=0.836 | n_tset=100.00, n_jw=100.00, a_tset=93.55, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=0.54, r_ncand=3.00
    * `S1-396207952` Interstate Realty Laboratories | 4602 Smithfield Drive, Northport, AL | US
    * `S2-584829505` Interstate Realty Laboratories Inc | 4623 SMITHFIELD DRIVE, NORHTPORT, AL | US
* p=0.967 | n_tset=100.00, n_jw=91.43, a_tset=96.08, num_jacc=0.50, code_first_eq=0.00, blk_frac_qmax=1.00, r_ncand=2.00
    * `S1-241383422` Rajputana Services (India) Private Limited | Anaghaiifloor, No.4, Devasandra, Newbelroad, Bangalore, Karnataka | India
    * `S3-801408223` Rajputana Services (India) Overseas Parivtae Limited | H.no 14 Anaghaiifloor, No.4, Devasandra, Newbelroad, Banglore, KA | India
* p=0.892 | n_tset=100.00, n_jw=0.00, a_tset=96.82, num_jacc=0.75, code_first_eq=0.00, blk_frac_qmax=1.00, r_ncand=10.00
    * `S1-513212777` SA Gold Private Limited | Divya Diamonds House, 3-225, 5Th Floor, Kavuri Hills Road, Cbi Colony, Madhapur, Hyderabad, Rangareddy, Telangana | India
    * `S2-132857833` ESA Gold Private Limited | NO 8 DIVYA DIAMONDS HOUSE, 3-225, 5TH FLOOR, KAVURI HILLS ROAD, CBI COLONY, MADHAPUR, HYDERABAD, Andhra Pradesh | India
* p=0.996 | n_tset=100.00, n_jw=100.00, a_tset=92.59, num_jacc=0.33, code_first_eq=0.00, blk_frac_qmax=0.70, r_ncand=1.00
    * `S1-752860459` Raum Holdings | 99 Kent Street, Unit 7501, Brookline, MA | US
    * `S3-904000073` Raum Holdings | 101 Kent Street, # 7501, Brookline, Massachusetts | US
* p=0.997 | n_tset=81.25, n_jw=91.09, a_tset=96.55, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=0.28, r_ncand=2.00
    * `S1-620559980` Vanguard Consumer Corporation | 234 Mulvihill Avenue, Redlands, CA | US
    * `S3-269111935` Vanguard Corporation Corp Center | 236 Mulvihill Avenue, Redlands, California | US
* p=0.948 | n_tset=69.57, n_jw=67.12, a_tset=96.55, num_jacc=0.50, code_first_eq=0.00, blk_frac_qmax=0.64, r_ncand=5.00
    * `S1-873818123` Devaya Sadashiv Private Limited | Kishan Gate, Lodhika Gidc, Kalawad Road, Metoda, Gujarat, Lodhika, Rajkot, Plot No. G-2115 | India
    * `S2-190975523` Devaex 5adashiv-Private Limited | www.devaexad.com | PLOT 410 PLOT NO. G-2115, KISHAN GATE, LODHIKA GIDC, KALAWAD ROAD, METODA, LODHIKA, Gujarat | India

## Matcher false negatives (true candidate rejected)

* p=0.209 | n_tset=100.00, n_jw=94.74, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.22, r_ncand=22.00
    * `S1-568295561` Nelson, Payne & Gargus LLC | 6 Pine View Road, New Castle, NY | US
    * `S2-155851058` NELSON GARGUS & PAYNE LLC |  | US
* p=0.045 | n_tset=82.76, n_jw=92.92, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.16, r_ncand=45.00
    * `S1-776861363` Metropolitan Dental Associates | 3039 Hazelton Street, Fairfax County, VA | US
    * `S3-197819440` Metropolitan Déntal Services |  | US
* p=0.127 | n_tset=28.57, n_jw=48.74, a_tset=100.00, num_jacc=0.50, code_first_eq=1.00, blk_frac_qmax=0.28, r_ncand=2.00
    * `S1-262983118` Porter Veterinary Clinic LLC | 100 Briarhaven Drive, Unit 105, Granite City, IL | US
    * `S3-139849679` Aviavilum | ##100 Briarhaven Drive, Granite City, Illinois | US
* p=0.029 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.10, r_ncand=54.00
    * `S1-288437407` Silver Producer Pvt Ltd | Shop No. 163, V. J. Patel Vegetable Market, Deesa, Banas Kantha, Gujarat | India
    * `S3-53532069` Silver Producer  Pvt Ltd |  | India
* p=0.769 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.47, r_ncand=34.00
    * `S1-768582961` Fleurette Hayes Return PLLC | Stoughton, MA, 646 West Street | US
    * `S2-577553854` FLEURETTE HAYES RETURN |  | US
* p=0.010 | n_tset=100.00, n_jw=100.00, a_tset=92.31, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=0.62, r_ncand=11.00
    * `S1-315398695` Good Tankers Private Limited | Door No 55/321, Ponneth Temple Road, Ernakulam, Kerala | India
    * `S2-399662505` GOOD TANKERS (PRIVATE) | 5561, PONNETH TEMPLE ROAD, ERNAKULAM, Kerala | India
* p=0.194 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.39, r_ncand=80.00
    * `S1-246487769` Sadyne Premier Dynamix LLC | 158 Ennis Road, Atkins, AR | US
    * `S3-33670676` Sadyne Premier Dynamix |  | US
* p=0.579 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.47, r_ncand=82.00
    * `S1-791675570` Ivenari Physical | 10819 300, Dunkirk, IN | US
    * `S3-573024878` The Ivenari  Physical |  | US
* p=0.602 | n_tset=21.43, n_jw=50.40, a_tset=94.74, num_jacc=0.33, code_first_eq=0.00, blk_frac_qmax=0.14, r_ncand=1.00
    * `S1-385603325` Smith and Sheridan Steers LLC | Palmyra, MO, 2176 321 | US
    * `S3-447211119` Ectosyn | 8176 321, Palmyra, Missouri | US
* p=0.051 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.23, r_ncand=82.00
    * `S1-580380821` Anjani Brothers Pvt. Ltd. | City Palace Mother Hospital Road Ambedkar Chauraha, Civil Line, Pratapgarh, Uttar Pradesh | India
    * `S2-39909432` Smt Anjani Brothers Pvt. |  | India
* p=0.417 | n_tset=100.00, n_jw=100.00, a_tset=92.86, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=0.56, r_ncand=5.00
    * `S1-716327974` Premier Engineering Enterprises | Fl 0, KS, Wichita, 714 Golden Hills Street | US
    * `S2-385338134` PREMIER ENGINEERING ENTERPRISES LP | WICHITA, 553  GOLDEN HILLS STREET, KS | US
* p=0.079 | n_tset=100.00, n_jw=100.00, a_tset=95.65, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=1.00, r_ncand=22.00
    * `S1-216135643` Orthopedic Care | Newark, DE, 215 Vassar Drive | US
    * `S2-767414136` Orthopedic Cáre Ltd | 211 VASSAR DR, NEWARK, DE | US
* p=0.031 | n_tset=100.00, n_jw=91.82, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.10, r_ncand=66.00
    * `S1-685337304` Hari Products Private Limited | G-10, Saraswati Enclacve Pataudi Road, Gurgaon, Haryana | India
    * `S2-821472349` Hari Products Praivfte Limited |  | India
* p=0.731 | n_tset=100.00, n_jw=91.11, a_tset=95.35, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=0.55, r_ncand=1.00
    * `S1-664894259` Green Care Private Limited | Mumbai, 108/B Evening Gloryraheja Vihar Powai, Maharashtra, Mumbai | India
    * `S2-583453885` Green Care Praivte Limited | H.NO 144 EVENING GLORYRAHEJA VIHAR POWAI, MUMBAI, Maharashtra | India
* p=0.246 | n_tset=100.00, n_jw=92.17, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.08, r_ncand=18.00
    * `S1-811674061` Smart National Couriers LLC | IL, 315 Arabian Circle, Willowbrook | US
    * `S2-441384879` Smart (National) |  | US
* p=0.619 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.27, r_ncand=7.00
    * `S1-230207575` Hernandez Ubs LLC | 5240 Pike Creek Lane, Indianapolis, IN | US
    * `S3-481706076` Hernandez Hernandez Ubs Llc |  | US
* p=0.290 | n_tset=31.37, n_jw=56.15, a_tset=100.00, num_jacc=0.50, code_first_eq=0.00, blk_frac_qmax=0.19, r_ncand=2.00
    * `S1-101923707` Behavioral Health Reliable Physicians | 1713 County Rd 609, TN, Etowah | US
    * `S2-98015471` Calolumriza | COUNTY RD 609, ETOWAH, TN | US
* p=0.015 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.39, r_ncand=59.00
    * `S1-209799606` Aditya Logistics Limited | Chennai, Plot No.17, Sri Ponniamman Nagar, Manali New Town, Tamil Nadu, Chennai | India
    * `S3-899415178` Aditya Logistics Ltd |  | India
* p=0.558 | n_tset=100.00, n_jw=95.79, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.24, r_ncand=29.00
    * `S1-96106537` Rogers Atlantic Usa | 702 Mulberry Street, Creston, IA | US
    * `S2-111745513` ROGERS ATLANTIC |  | US
* p=0.206 | n_tset=100.00, n_jw=94.36, a_tset=77.27, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=0.84, r_ncand=5.00
    * `S1-598996638` Mesa Pediatric Dental Health | 542 Higley Road, Unit 45, Mesa, AZ | US
    * `S2-922864241` MESA PEDIATRIC DENTAL HEALTH COMMISSION | 40 HIGLEY ROAD, MESA DESERT, AZ | US
* p=0.291 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.31, r_ncand=63.00
    * `S1-760730060` Surgical Silver Specialists LLC | 58 Gettysburg Avenue, Tonawanda, NY | US
    * `S3-820715500` Surgical Silver Specialists LLC |  | US
* p=0.174 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.48, r_ncand=55.00
    * `S1-862683477` Nari Tello Holding PLLC | 105-16 K, Brooklyn, NY | US
    * `S2-2057003` Nari Tello Holding |  | US
* p=0.015 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.19, r_ncand=64.00
    * `S1-942596512` Downtown Massage | 5464 Riverview Circle Drive, Thomson, IL | US
    * `S2-305234336` Downtown Mássage |  | US
* p=0.081 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.32, r_ncand=74.00
    * `S1-666721068` Peak Equity Alliance Inc | Bldg BUILDING, Park City (summit Co), 1345 Pinnacle Drive, UT | US
    * `S3-601245322` Peak Equity Alliance |  | US
* p=0.574 | n_tset=88.46, n_jw=95.26, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.18, r_ncand=14.00
    * `S1-344114550` Latisha Garrison Portfolio LLC | 1026 5th Street, Washington, DC | US
    * `S2-816393160` LATISHA GATIRSMO PORTFOLIO [LLC] |  | US
