# Error analysis (out-of-fold predictions, full training set)

* true pairs: 7,638,365
* missed by blocking (never a candidate): 166,686 (2.18%)
* missed by the matcher (candidate rejected): 184,312 (2.41%)
* wrong merges (false positive pairs): 17,385 of 7,304,752 predicted (0.24%); on singleton S1s: 2,344

| slice | S1 entities | macro F0.5 |
|---|---|---|
| India | 883,188 | 0.9783 |
| US | 1,323,633 | 0.9845 |
| singletons | 123,247 | 0.9833 |
| 1 true match | 119,157 | 0.9405 |
| 2-4 true matches | 1,390,168 | 0.9832 |
| 5+ true matches | 574,249 | 0.9874 |

## Candidate traits: share among all true candidate pairs vs among errors

| trait | all true pairs | matcher misses (FN) | wrong merges (FP) |
|---|---|---|---|
| empty_address | 4.1% | 70.9% | 21.8% |
| indic_name | 6.8% | 1.1% | 3.3% |
| domain_name | 4.7% | 0.5% | 1.2% |
| fka_alias | 2.0% | 0.0% | 0.0% |
| name_sim<60 | 6.4% | 8.6% | 15.6% |
| first_code_differs | 20.2% | 19.6% | 49.3% |

## False positives on singleton S1s (each costs a full 1.0)

* p=0.965 | n_tset=97.44, n_jw=99.00, a_tset=100.00, num_jacc=1.00, code_first_eq=1.00, blk_frac_qmax=1.00, r_ncand=3.00
    * `S1-454540138` Empire Alliance III | 555 Pride Avenue, Auburn, AL | US
    * `S3-152464477` Empire  Alliance FIII | 555 Pride Ave, Auburn, Alabama | US
* p=0.692 | n_tset=100.00, n_jw=100.00, a_tset=100.00, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.27, r_ncand=2.00
    * `S1-398920347` Lotus Logistics Pvt Ltd | No. 42, 2Nd Cross, Exservicemen Colony, Bangalore, 3Rd Floor, Karnataka, Bangalore North | India
    * `S3-849876847` Lotus Logistics | #, Bangalore, KA | India
* p=0.706 | n_tset=91.43, n_jw=88.80, a_tset=80.00, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=0.16, r_ncand=2.00
    * `S1-777016531` Prime Investments Private Limited | 1St Floor, Survey No 221, Goregoan Mulund Link Road, Bhandup, Mumbai, Mumbai City, Maharashtra | India
    * `S2-451724144` प्राइम इन्वेस्टमेंट्स प्राइवेट लिमिटेड | FLAT NO 05, MUMBAI, MUMBAI CITY, Maharashtra | India
* p=0.719 | n_tset=100.00, n_jw=100.00, a_tset=94.12, num_jacc=0.33, code_first_eq=0.00, blk_frac_qmax=1.00, r_ncand=3.00
    * `S1-823842386` Covington Greenfire | Bldg 5, OH, Galloway, 1119 Burlake Square | US
    * `S3-108579220` *** covington greenfire ltd | 1130 Burlake Square, Building 5, Galloway, Ohio | US
* p=0.510 | n_tset=100.00, n_jw=100.00, a_tset=92.31, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=1.00, r_ncand=14.00
    * `S1-477656862` Blue Stellar | 5247 Ardrey Kell Road, Charlotte, NC | US
    * `S3-399048539` Blue Stellar Inc | 5268b Ardrey Kell Rd, North Carolina, Charlotte | US
* p=0.644 | n_tset=100.00, n_jw=100.00, a_tset=86.11, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=1.00, r_ncand=10.00
    * `S1-631727754` Ryan Holdings Group | 3518 Valley Ridge Road, City Of Middleton, WI | US
    * `S2-282284505` Ryan Holdings Group Holdings | 3520 VALLEY RIDGE RD, MIDDLETTON, WI | US
* p=0.968 | n_tset=100.00, n_jw=100.00, a_tset=92.06, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=1.00, r_ncand=7.00
    * `S1-984926415` Regional Seafood Inc. | 2085 Aldersgate Road, Rock Hill, SC | US
    * `S2-771731937` Regional Seafood Inc | 2090A ALDERSGATE RD, ROCK HILL, SC | US
* p=0.842 | n_tset=94.44, n_jw=93.07, a_tset=100.00, num_jacc=1.00, code_first_eq=1.00, blk_frac_qmax=1.00, r_ncand=2.00
    * `S1-829322124` Prairie Council IV | 12449 Tejas Court, Rancho Cucamonga, CA | US
    * `S3-337646399` Prairie UV Council | 12449 Tejas Ct, Rancho Cucamonga, California | US
* p=0.647 | n_tset=100.00, n_jw=100.00, a_tset=100.00, num_jacc=1.00, code_first_eq=1.00, blk_frac_qmax=1.00, r_ncand=43.00
    * `S1-888041196` Orthopedic Health P.C. | 7467 Woodhaven Drive, La Plata, MD | US
    * `S3-414755944` Co Orthopedic Health | 7467 Woodhaven Dr, La Plata, Maryland | US
* p=0.987 | n_tset=100.00, n_jw=77.58, a_tset=100.00, num_jacc=1.00, code_first_eq=1.00, blk_frac_qmax=1.00, r_ncand=4.00
    * `S1-453076238` Elita Ramirez Hovnanian Inc | 209 Norris Street, Burlington, WA | US
    * `S3-71745577` Ramirez Ramirez Hovnanian | 209 Norris St, Burlington, Washington | US
* p=0.843 | n_tset=100.00, n_jw=100.00, a_tset=68.66, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=1.00, r_ncand=6.00
    * `S1-618541929` Physical Therapy Group | 2052 Montreat Circle, Vestavia Hills, AL | US
    * `S3-759176797` Physical Therapy Group | 2055 Montreat Cir, Birmiingham, Alabama | US
* p=0.860 | n_tset=100.00, n_jw=100.00, a_tset=94.59, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=1.00, r_ncand=17.00
    * `S1-721849204` Vetasoft Carolina Inc. | 1643 Castle Creek Drive, Missouri City, TX | US
    * `S2-367794656` Inc Vetasoft Carolina | 1650 CASTLE CREEK DRIVE, MISSOURI CITY, TX | US
* p=0.587 | n_tset=100.00, n_jw=100.00, a_tset=86.27, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=1.00, r_ncand=6.00
    * `S1-29155276` Yang Managers | 4201 Maple Avenue, Lyons, IL | US
    * `S3-364539112` LLC Yang Managers | 4206 Maple Avenue, LYON Scity, Illinois | US
* p=0.888 | n_tset=97.30, n_jw=98.60, a_tset=87.50, num_jacc=1.00, code_first_eq=1.00, blk_frac_qmax=0.95, r_ncand=1.00
    * `S1-418544007` XRD Heartland Aria, LLC | VA, 150 Stallion Court, Amherst County | US
    * `S2-280374228` XRCD Heartland Aria, LLC | 150- STALLION CT, AMHERSST CDP, VA | US
* p=0.762 | n_tset=100.00, n_jw=100.00, a_tset=96.77, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=1.00, r_ncand=12.00
    * `S1-980858330` Cassaundra's Bakery | 1902 Porter Street, Richmond City, VA | US
    * `S2-765033246` Cassaundra's Bakery Llc | 1905 PORTER ST, VA, RICHMOND CITY | US
* p=0.974 | n_tset=100.00, n_jw=100.00, a_tset=80.00, num_jacc=1.00, code_first_eq=1.00, blk_frac_qmax=1.00, r_ncand=4.00
    * `S1-637297876` Midwest Infrastructure LLC | 1086 25th Place, Pleasantville, IA | US
    * `S3-831711755` Midwest-Infrastructure  Corp | 1086 25th Pl, Pleasant Grove, Iowa | US
* p=0.477 | n_tset=100.00, n_jw=91.00, a_tset=82.76, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=0.41, r_ncand=3.00
    * `S1-106304` First Power Private Limited | No. 161-D, 3Rd Floor, Ag-1, Vikaspuri, New Delhi, Delhi | India
    * `S3-422683172` first power holdings private limited | No 164-D, New Delhi, Vikaspuri, DL | India
* p=0.583 | n_tset=93.33, n_jw=97.33, a_tset=92.86, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=0.70, r_ncand=7.00
    * `S1-4944019` Hailey, Hanny, DO | 140 Sand Creek Road, Colonie, NY | US
    * `S3-307470991` Hailey, Hanny, DH | 144 Snad Creek Rd, Colonie, New York | US
* p=0.803 | n_tset=75.00, n_jw=86.89, a_tset=96.00, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=0.43, r_ncand=5.00
    * `S1-558019775` Chandrora Software | B 106, Ground Floor, Shantishikara Apts Somajiguda, Hyderabad, Kurnool, Andhra Pradesh | India
    * `S2-178857179` Chandrora India | ఆంధ్రప్రదేశ్, B 119, GROUND FLOOR, SHANTISHIKARA APTS SOMAJIGUDA, HYDERABAD | India
* p=0.994 | n_tset=100.00, n_jw=94.40, a_tset=93.33, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=0.78, r_ncand=1.00
    * `S1-259186067` Shyam Technologies Private Limited | Sb-86, Sector-63A, Fng Road, Noida, Gautam Buddha Nagar, Uttar Pradesh | India
    * `S3-770296549` Shyam Technologies प्रा. लि. | ##173, Noida, Gautam Buddha Nagar, UP | India
* p=0.956 | n_tset=100.00, n_jw=100.00, a_tset=85.25, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=0.88, r_ncand=7.00
    * `S1-202834513` Kerby's Yoga | Asheville, 3 Bear Creek Lane, Unit A, NC | US
    * `S2-565332001` Kerby's Yoga LLC | ASHEVILLE, 8 BEAR CREEK LANE, NC, PMB 5118 | US
* p=0.565 | n_tset=16.00, n_jw=38.46, a_tset=100.00, num_jacc=1.00, code_first_eq=1.00, blk_frac_qmax=1.00, r_ncand=25.00
    * `S1-302730481` Global Finance Private Limited | Level 9, World Trade Center, Tower 2, The Executive Center, Near Eon It Park, Kharadi, Pune, Maharashtra | India
    * `S2-881948351` durai trust | LEVEL 9, WORLD TRADE CENTER, TOWER 2, THE EXECUTIVE CENTER, NEAR EON IT PARK, KHARADI, PUNE, Maharashtra | India
* p=0.837 | n_tset=100.00, n_jw=65.41, a_tset=91.18, num_jacc=0.11, code_first_eq=0.00, blk_frac_qmax=0.09, r_ncand=3.00
    * `S1-954697415` Kk Enterprises | Plot No. 347/A & 347/B, Trendz Uptown, 6Th Floor, Sy No. 33, 34 Part, 35 Part, 36, 38&39 Of Guttala Begumpet Village, Shaikpet, Hyderabad, Telangana | India
    * `S2-961570374` Holdings Enterprises Kk | PLOT NO. D/368/A & 347/B, SHAIKPET, HYDERABAD, Telangana | India
* p=0.773 | n_tset=59.57, n_jw=81.42, a_tset=98.78, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=0.69, r_ncand=1.00
    * `S1-4876320` Bright International Private Limited | 4Th Floor, Sugam Corporate House, Nr.Diva Hospital Parimal Garden, Navrangpura, Ahmedabad, Gujarat | India
    * `S3-449711088` બ્રાઇટ ઇન્ટરનેશનલ બેકરી પ્રાઇવેટ લિમિટેડ | 8Th Floor, Sugam Corporate House, Nr.diva Hospital Parimal Garden, Navrangpura, Ahmedabad, GJ | India
* p=0.946 | n_tset=100.00, n_jw=100.00, a_tset=95.24, num_jacc=0.33, code_first_eq=0.00, blk_frac_qmax=1.00, r_ncand=2.00
    * `S1-948658126` Joshi Holdings Group of Shipshewana | 3650 5, Shipshewana, IN | US
    * `S3-486019375` Joshi Holdings Group of Shipshewana | 3657 5, Shipshewana, Indiana | US

## False positives on non-singleton S1s

* p=0.815 | n_tset=94.74, n_jw=97.89, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.10, r_ncand=105.00
    * `S1-936839900` Spears Golden Orion Inc | 320 Glendare Drive, Unit D, Winston-salem, NC | US
    * `S3-430162260` Spears 6olden 6olden Orion Inc |  | US
* p=0.791 | n_tset=100.00, n_jw=100.00, a_tset=93.62, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=1.00, r_ncand=15.00
    * `S1-239301159` Chiropractic Gulf Care Associates L.L.C. | 712 Taft Road, Hinsdale, IL | US
    * `S3-230883824` Chiropractic Gulf Care Associates (LLC) | 716-B Taft Road, Hinsdale, Illinois | US
* p=0.999 | n_tset=97.56, n_jw=99.05, a_tset=100.00, num_jacc=0.50, code_first_eq=0.00, blk_frac_qmax=0.52, r_ncand=3.00
    * `S1-764724843` Solutions Sky Systems Ltd | 6 A 15 Jawahar Nagar, Sri Ganganagar, Ganganagar, Rajasthan | India
    * `S2-549184023` Solutions Sky System ((Ltd)) | #15 A 15 JAWAHAR NAGAR, SRI GANGANAGAR, Rajasthan | India
* p=0.979 | n_tset=100.00, n_jw=100.00, a_tset=95.65, num_jacc=0.33, code_first_eq=0.00, blk_frac_qmax=0.81, r_ncand=8.00
    * `S1-168795692` Lotaon Equity Partners Co | 17806 103rd Court, Redmond, WA | US
    * `S3-898341290` Lotaon Equity Partners Partners Co | 17809 103rd Court, null, Redmond, Washington | US
* p=0.814 | n_tset=100.00, n_jw=100.00, a_tset=87.80, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=0.71, r_ncand=5.00
    * `S1-149792200` American Landscaping L.L.C. | OK, Norman, 3412 Grant Road | US
    * `S3-357410855` American Landscaping LLC | 3433-3435 Grant Road, PO Box 7566, Norman, Oklahoma | US
* p=0.993 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.23, r_ncand=77.00
    * `S1-411414517` Ravika Mining | H/O Shakila Begum Makhdum Rasti Colony Minhaj Nagar Near Masjid Phulwari Sharif, Patna, Bihar | India
    * `S3-231580662` Ravika  Private Mining Limited |  | India
* p=1.000 | n_tset=100.00, n_jw=100.00, a_tset=100.00, num_jacc=1.00, code_first_eq=1.00, blk_frac_qmax=1.00, r_ncand=15.00
    * `S1-197010175` Corner Hypnosis | 16900 Salmonberry Road, Brookings, OR | US
    * `S2-453673792` Corner Hypnosis LLC | 16900 SALMONBERRY RD, BROOKINGS, OR | US
* p=0.995 | n_tset=100.00, n_jw=100.00, a_tset=90.57, num_jacc=1.00, code_first_eq=1.00, blk_frac_qmax=0.87, r_ncand=10.00
    * `S1-282727393` Perry, Gavra, O.D., M.D., P.C. | 3 Chloe Court, Bloomington, IL | US
    * `S2-605319011` Perry, Gavra, O.D., M.D., P.C. Inc | 3 CHLOE CT, BLOOINGTON CDP, IL | US
* p=0.915 | n_tset=100.00, n_jw=100.00, a_tset=98.06, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=1.00, r_ncand=2.00
    * `S1-703528589` Ksr Enterprises | No 15 Krishnanjali Anjanappa Layout Doddabommasandra, Vidyaranyapura, Bangalore North, Bangalore, Karnataka | India
    * `S2-982259600` Ksr Enterprises Limited | NO G-20 KRISHNANJALI ANJANAPPA LAYOUT DODDABOMMASANDRA, VIDYARANYAPURA, BANGALORE NORTH, Karnataka | India
* p=0.880 | n_tset=100.00, n_jw=100.00, a_tset=67.65, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=0.81, r_ncand=6.00
    * `S1-50913830` Ridgeline Urban Group | 959 Riverview Road, Clifton Park, NY | US
    * `S3-808664805` Ridgeline Urban Group | Rexofrd Township, 964 Riverview Rd, New York | US
* p=0.852 | n_tset=66.67, n_jw=86.69, a_tset=92.68, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=0.14, r_ncand=2.00
    * `S1-655125601` Curated Solutions LLP | Ground Floor, No.63/21, 2Nd, Cross, Govt School Rd, Bangalore South, Bangalore, Karnataka | India
    * `S3-704824782` Curated LLP Services | No 28 Ground Floor, Bangalore, Bangalore South, KA | India
* p=0.807 | n_tset=100.00, n_jw=100.00, a_tset=92.06, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=0.52, r_ncand=23.00
    * `S1-735303258` Optimal Analytics Union | 2146 Deep Blue Drive, UT, Unit 202, West Valley City | US
    * `S2-954747309` Optimal Analytics Union Inc | 2149 DEEP BLUE DRIVE, W VALLEY CITY, UT | US
* p=0.865 | n_tset=62.07, n_jw=72.70, a_tset=100.00, num_jacc=0.33, code_first_eq=1.00, blk_frac_qmax=0.36, r_ncand=2.00
    * `S1-788745507` Ralphs Wireless | 5501 Monalee Avenue, Sacramento, CA | US
    * `S3-130963201` Garcia Wgeless | 5501 1/2 Monalee Ave, Sacramento, California | US
* p=0.749 | n_tset=100.00, n_jw=89.30, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.09, r_ncand=303.00
    * `S1-48100072` Jhabua (India) India Private Limited | 145B/9, Ground Floor, Near Akhara Kishangarh, Vasant Kunj, New Delhi, South West Delhi, Delhi | India
    * `S3-49877548` Jhabua Global(India) Private Limited |  | India
* p=0.797 | n_tset=100.00, n_jw=100.00, a_tset=91.23, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=1.00, r_ncand=9.00
    * `S1-496225005` 4761 Third Lane Owners Corp | 1918 Falls Valley Drive, Raleigh, NC | US
    * `S3-964514744` 4761 Third Lane Owners Corp | North Carolina, Raleigh, 1925-1927 Falls Valley Dr | US
* p=0.918 | n_tset=100.00, n_jw=100.00, a_tset=100.00, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=1.00, r_ncand=5.00
    * `S1-294993950` Nanda Holdings | Village - Manani, Post Nalhera Gujjar Teh- Saharanpur, Saharanpur, Uttar Pradesh | India
    * `S3-838250623` Nanda Holdings LLP | No 33 Village - Manani, Saharanpur, Post Nalhera Gujjar Teh- Saharanpur, UP | India
* p=0.869 | n_tset=100.00, n_jw=100.00, a_tset=96.43, num_jacc=0.33, code_first_eq=0.00, blk_frac_qmax=0.70, r_ncand=5.00
    * `S1-442666538` NM Art Inc. | CA, 25094 3rd Street, San Bernardino | US
    * `S3-896347184` NM ART INC | #25095 3rd Street, San Bernardino, California | US
* p=0.969 | n_tset=100.00, n_jw=100.00, a_tset=98.88, num_jacc=0.50, code_first_eq=0.00, blk_frac_qmax=0.60, r_ncand=38.00
    * `S1-248344281` Vision Systems Private Limited | Maharashtra, Ahmednagar, Plot No A- 47/4 Midc Industrial Area | India
    * `S2-746103065` Visi0n Systems Limited | PLOT NO A- 47/ MIDC INDUSTRIAL AREA, AHMEDNAGAR, महाराष्ट्र | India
* p=0.807 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.69, r_ncand=63.00
    * `S1-987788053` Hollenbeck Reliable Joint LLC | 55 Fox Street, Hubbard, OH | US
    * `S3-240504489` HOLLENBECK RELIABLE JOINT LLC |  | US
* p=0.787 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.52, r_ncand=34.00
    * `S1-147351384` Colline Riley Integrated LLC | 2814 Timber Hills Rd, Louisville, KY | US
    * `S3-379469312` Colline-Riley Integrated |  | US
* p=0.884 | n_tset=100.00, n_jw=100.00, a_tset=98.51, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=1.00, r_ncand=14.00
    * `S1-454455909` Salter Permian | 7442 Kennedy Road, Fauquier County, VA | US
    * `S3-253185` Salter-Permian Ltd | 744 Kennedy Rd, Fauquier County, Virginia | US
* p=0.878 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.29, r_ncand=22.00
    * `S1-986748282` A Golden Kgaa | 10504 Gentlewind Ct, Louisville, KY | US
    * `S2-467281635` Golden Kgaa Pc |  | US
* p=0.785 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.42, r_ncand=169.00
    * `S1-9063002` Zephyx Rmg | 101 Savannah Terrace Drive, Wentzville, MO | US
    * `S3-116188018` Zephyx Rmg Ínc |  | US
* p=0.996 | n_tset=88.89, n_jw=91.78, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.24, r_ncand=105.00
    * `S1-785348670` Candida L. Stanford, DPM | 7762 La Fiesta Way, Sacramento, CA | US
    * `S3-772047142` Candida L. L. Stanford, Center |  | US
* p=0.996 | n_tset=100.00, n_jw=92.31, a_tset=93.51, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=0.55, r_ncand=6.00
    * `S1-761850089` Mansarovar Group of Companies | 3Rd Floor, K. K. Chambers, Sir P. T. Marg, Fort, Mumbai, Maharashtra | India
    * `S3-281652658` Mansarovar Group of | Mumbai, K. K. Chambers, Sir P. T. Marg, Door No ##8Rd Floor, महाराष्ट्र, Mumbai Ii | India

## Matcher false negatives (true candidate rejected)

* p=0.711 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.19, r_ncand=59.00
    * `S1-56076712` Jvs & Co | No. 1605, Embassy Habitat Palace Road, Bangalore, Karnataka | India
    * `S3-245910836` Jvs-& Có |  | India
* p=0.601 | n_tset=97.56, n_jw=99.05, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.37, r_ncand=34.00
    * `S1-254104863` 106-107 North Owners Corp | 57 Story Street, Boston, MA | US
    * `S2-246430586` 106-107 North Owlners Corp |  | US
* p=0.596 | n_tset=45.16, n_jw=55.57, a_tset=100.00, num_jacc=0.80, code_first_eq=0.00, blk_frac_qmax=0.67, r_ncand=11.00
    * `S1-172476221` Shantilal Traders LLP | 601, Floor-06, Plot-336, 336, 1, Mahjabeen Heights, Shaikh Misree Road, Mumbai, Maharashtra | India
    * `S3-777459810` Veragildaria | H.no 223 601, Floor-06, Plot-336, 336, 1, Mahjabeen Heights, Shaikh Misree Road, Mumbai, MH | India
* p=0.695 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.33, r_ncand=76.00
    * `S1-81568365` Value Aerospace Associates | 777 Timberline Parkway, Valparaiso, IN | US
    * `S3-465418937` Value Aerospace Associates Corp |  | US
* p=0.074 | n_tset=100.00, n_jw=100.00, a_tset=91.30, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=0.73, r_ncand=7.00
    * `S1-633195828` South Frontier Cbre Inc | 15602 Giese Lane, Manor, TX | US
    * `S2-434635304` SOUTH FRONTIER CBRE | 15221 GIESE LANE, MANOR, TX | US
* p=0.083 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.30, r_ncand=203.00
    * `S1-663732869` Motech Inc. | 1520 Wheeler Avenue, Albuquerque, NM | US
    * `S2-189132459` Motech Inc. |  | US
* p=0.528 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.42, r_ncand=23.00
    * `S1-890686728` RE Signature Sunstone | 16 Short Beach Road, North Haven, NY | US
    * `S2-807103335` re signature sunstone |  | US
* p=0.481 | n_tset=96.30, n_jw=98.10, a_tset=91.76, num_jacc=1.00, code_first_eq=1.00, blk_frac_qmax=0.83, r_ncand=7.00
    * `S1-342885919` Lily Holdings | 2Nd Floor, 11 Dr U.N. Brahmachari Street, Kolkata, Kolkata, Howrah, West Bengal | India
    * `S2-723100913` Liily Holdings | 2ND FLOOR, 11 DR U.N. BRAHMACHARI STREET, KOLKATA, KOLKATA, KOLKATTA, পশ্চিমবঙ্গ | India
* p=0.447 | n_tset=100.00, n_jw=100.00, a_tset=97.78, num_jacc=0.33, code_first_eq=0.00, blk_frac_qmax=0.49, r_ncand=23.00
    * `S1-251226893` Bangalore South (india) Private Limited | 007 2Nd Floor A Block Brundavan Garden, Kanakapura Main Road, Doddakalsandra, Bangalore South, Bangalore, Karnataka | India
    * `S3-678457044` Bangalore South (india) | 00 2Nd Floor A Block Brundavan Garden, Bangalore, Bangalore South, KA | India
* p=0.429 | n_tset=100.00, n_jw=100.00, a_tset=91.80, num_jacc=0.00, code_first_eq=0.00, blk_frac_qmax=0.80, r_ncand=5.00
    * `S1-531775679` Hatti Byfield Value Martin LLC | TN, 8085 Bell Campground Road, Powell | US
    * `S3-278186741` Hatti Byfield Value [Martin] | 7937 Bell Campground Road, Powell, Tennessee | US
* p=0.055 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.25, r_ncand=63.00
    * `S1-346308358` Desert Group | 465 Olmstead Street, Winona, MN | US
    * `S2-979464520` Desert-Group |  | US
* p=0.367 | n_tset=78.95, n_jw=86.21, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.11, r_ncand=50.00
    * `S1-38724531` Abdullah Medical Center | 119 County Road 499, Mexia, TX | US
    * `S3-845295479` Abdullah Center Partners |  | US
* p=0.036 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.28, r_ncand=51.00
    * `S1-149216889` Metropolitan Foundation | 744 Washington Street, Boston, MA | US
    * `S2-732605508` METROPOLITAN  FOUNDATION LP |  | US
* p=0.119 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.50, r_ncand=16.00
    * `S1-560853131` Patriot Investments | 7835 118th Street, Seattle, WA | US
    * `S2-108541462` PATRIOT ÍNVESTMENTS |  | US
* p=0.503 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.25, r_ncand=26.00
    * `S1-749420821` A Custom Dime LLC | 247 Broadway Road, Unit 34, Dracut, MA | US
    * `S2-883652900` A Custom Dime Llc |  | US
* p=0.476 | n_tset=93.02, n_jw=98.14, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.23, r_ncand=33.00
    * `S1-618939473` Garhwal Digital (India) Limited | C -225 Sector 48 Noida, Noida, Gautam Buddha Nagar, Uttar Pradesh | India
    * `S3-644280531` Garhwal Digital (Inda)i Limited |  | India
* p=0.003 | n_tset=34.29, n_jw=55.23, a_tset=90.00, num_jacc=0.33, code_first_eq=0.00, blk_frac_qmax=0.13, r_ncand=7.00
    * `S1-927796239` Anchor Freight Studios PLLC | 1019 Route 1, Steuben, ME | US
    * `S2-110263335` Brixsynfaye | 794  ROUTE 1, ME, STEUBEN | US
* p=0.252 | n_tset=27.78, n_jw=49.74, a_tset=100.00, num_jacc=0.50, code_first_eq=1.00, blk_frac_qmax=0.15, r_ncand=3.00
    * `S1-65266904` Laferriere, Hale & Vargas Era LLC | 13122 Vista Station Boulevard, Unit A322, Draper City (sl Co), UT | US
    * `S2-269968542` Tavodova | 13122 VISTA STATION BOULEVARD, null, DRAPER, UT | US
* p=0.021 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.23, r_ncand=65.00
    * `S1-321748750` 1373 Third Avenue Apartments P.C. | 3420 Kings Neck Drive, Virginia Beach City, VA | US
    * `S3-838096342` 1373 Third Avenue Apartments LLC |  | US
* p=0.184 | n_tset=92.75, n_jw=94.59, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.08, r_ncand=83.00
    * `S1-69770499` Ear Nose & Throat Rocky Care Associates LLC | 3700 Williams Field Road, Unit 3144, Gilbert, AZ | US
    * `S3-240032859` Ear Nose + Throat Rocky Associates LLC Center |  | US
* p=0.297 | n_tset=77.42, n_jw=88.09, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.22, r_ncand=19.00
    * `S1-152059002` All Learning Worldwide LLC | 3135 23rd Street, Brownsville, TX | US
    * `S3-221943603` All Learning LLC Center |  | US
* p=0.086 | n_tset=83.72, n_jw=92.21, a_tset=96.67, num_jacc=1.00, code_first_eq=1.00, blk_frac_qmax=0.33, r_ncand=8.00
    * `S1-685950139` Atlas Bond-Chattanooga | 9188 Bombora Lane, Chattanooga, TN | US
    * `S3-671941296` Atlas Bocg-Chatatnoga | 9188 Bobmora Lane, Chattanooga, Tennessee | US
* p=0.018 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.06, r_ncand=88.00
    * `S1-506554821` Hotel Hospitality Limited | Shop No.F4, Door No. Thiruvail 3-71/G24 First Floor City Max Building Vamanjaoor, Thiruvail, Mangalore, Dakshina Kannada, Karnataka | India
    * `S2-504667810` HOTEL HOSPITALITY |  | India
* p=0.555 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.43, r_ncand=97.00
    * `S1-562789585` Madden Entertainment | 1005 Strock Street, Blacksburg Town, VA | US
    * `S3-789051230` Madden Entertainment |  | US
* p=0.017 | n_tset=100.00, n_jw=100.00, a_tset=nan, num_jacc=nan, code_first_eq=nan, blk_frac_qmax=0.26, r_ncand=332.00
    * `S1-421069309` Shiva Consulting Pvt Ltd | D-127 Ramprastha Ghaziabad, Ghaziabad, Uttar Pradesh | India
    * `S2-322892106` Shiva Consulting Pvt Ltd. |  | India
