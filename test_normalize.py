"""Normalisation tests built from real noise patterns observed in the training data."""
from ber.normalize import normalize_address, normalize_name, phonetic_key
from ber.translit import transliterate


def core(s):
    return normalize_name(s).core


def test_legal_suffix_variants_share_core():
    variants = ["Herrera Family Office LLC", "HERRERA FAMILY OFFICE (LLC)", "Herrera  Family Office [Llc]",
                "LLC Herrera Family Office", "Herrera Family Office L.L.C."]
    assert {core(v) for v in variants} == {"herrera family office"}


def test_leetspeak_accents_and_junk_prefixes():
    assert core("Wils0n Averin") == "wilson averin"
    assert core("*** Jo Ann's  Prairie Fámily Practice LP") == "jo anns prairie family practice"
    assert core("5bl Entertainment Entertainment Public") == "sbl entertainment"   # dup word dropped


def test_alias_and_domain_names():
    n = normalize_name("Vantagedova fka Pioneer Redwood L.L.C.")
    assert n.core == "vantagedova" and n.alias == "pioneer redwood"
    n = normalize_name("Belokor F/K/A Houston Veterans Ridge Federation")
    assert n.alias == "houston veterans ridge federation"
    for raw, expect in [("schroederprinting.com", "schroederprinting"),
                        ("pediatricdentistryphysicianscom", "pediatricdentistryphysicians"),
                        ("Zephyr.Com", "zephyr")]:
        n = normalize_name(raw)
        assert n.is_domain and n.core == expect


def test_indic_transliteration_matches_latin_core():
    assert transliterate("बेस्ट फूड्स प्राइवेट लिमिटेड") == "best phuds praivet limited"
    assert core("बेस्ट फूड्स प्राइवेट लिमिटेड") == "best phuds"           # legal words recognised
    assert core("ശക്തി ഇംപെക്സ് പ്രൈവറ്റ് ലിമിറ്റഡ്") == "shakti impeks"   # Malayalam
    assert normalize_name("राम मार्केटिंग प्राइवेट लिमिटेड").is_indic


def test_phonetic_key_bridges_scripts():
    assert phonetic_key("private") == phonetic_key("praivet") == phonetic_key("piraivet")
    assert phonetic_key("global") == phonetic_key("kulopal")      # Tamil has no voicing marks
    assert phonetic_key("limited") == phonetic_key("limittad")
    assert phonetic_key("1234") == "1234"


def test_french_legal_forms():
    assert normalize_name("Gagny (France) Club (S.A.R.L.)").legal == "sarl"
    assert normalize_name("S.A.S. Connect Groupement France").core == "connect groupement france"


def test_address_reordering_and_abbreviations_converge():
    a = normalize_address("W147N10734 Heritage Parkway, Village Of Germantown, WI", "US")
    b = normalize_address("VILLAGE OF GERMANTOWN, W147N10734 HERITAGE PKWY, WI", "US")
    assert sorted(a.norm.split()) == sorted(b.norm.split())
    assert a.state == b.state == "wi"


def test_street_saint_corruption_and_state_names():
    a = normalize_address("17 Longhill Street, Springfield, MA", "US")
    b = normalize_address("17 Longhill Saint, Springfield, Massachusetts", "US")
    assert a.norm == b.norm


def test_state_codes_that_are_stopwords_still_detected():
    assert normalize_address("Denver, CO, 12 Main St", "US").state == "co"
    assert normalize_address("Baton Rouge, LA", "US").state == "la"


def test_india_native_script_state_and_house_numbers():
    a = normalize_address("726 GF SECTOR 23, GURUGRAM, हरियाणा", "India")
    b = normalize_address("Haryana, DOOR NO 726 GF SECTOR 23, GURUGRAM", "India")
    assert a.state == b.state == "hr"
    assert a.nums == ["726", "23"] and b.nums == ["726", "23"]
    assert normalize_address("x, தமிழ்நாடு", "India").state == "tn"


def test_unit_numbers_distinguish_siblings():
    a = normalize_address("D.No.6-62-C6, Udupi, Karnataka", "India")
    b = normalize_address("D.no.6-62-c11, Karkala, Udupi, KA", "India")
    assert a.codes[0] == "662c6" and b.codes[0] == "662c11"


def test_fillers_and_empty():
    assert normalize_address("6505 SHAFTSBURY DR, N/A, KNOXVILLE, TN", "US").norm == "6505 shaftsbury dr knoxville tn"
    assert normalize_address("", "US").empty


def test_french_overlay_is_country_scoped():
    fr = normalize_address("N° 50 R. DE LA BENAUGE, BORDEAUX, Nouvelle-Aquitaine", "France")
    assert fr.norm == "50 rue benauge bordeaux naq"
    india = normalize_address("168 R Gopalapuram, Pollachi", "India")
    assert "rue" not in india.norm.split()


def test_unknown_country_uses_generic_rules():
    x = normalize_address("12 Main Street, Springfield", "Atlantis")
    assert x.norm == "12 main st springfield" and x.state == ""


def test_segmenter_splits_concatenated_names_only_when_fully_covered():
    from ber.normalize import Segmenter
    seg = Segmenter({"sears": 5, "platforms": 3, "zfj": 2, "club": 40, "innovative": 30, "brothers": 50})
    assert normalize_name("searsplatforms", seg).core == "sears platforms"
    n = normalize_name("ZFJCLUBSARL", seg)
    assert n.core == "zfj club" and n.legal == "sarl"            # trailing legal suffix separated
    assert normalize_name("Vantageriza", seg).core == "vantageriza"   # not coverable -> untouched
    assert normalize_name("Innovative Brothers", seg).core == "innovative brothers"


def test_zero_runs_inside_codes_are_normalised():
    assert normalize_address("L053 Kaniyara Rd, Mysore, KA", "India").codes == ["l53"]
    assert normalize_address("L53 Kaniyara Rd, Mysore, KA", "India").codes == ["l53"]
    assert normalize_address("0 Main St", "US").codes == ["0"]
