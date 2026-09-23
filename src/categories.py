
attribute_category_names_map = {
    'economic__education': 'education',
    'economic__employment_status': 'employment status',
    'economic__income_wealth_economic_status': 'income/wealth/economic status',
    'economic__occupation_profession': 'occupation/profession',

    'noneconomic__age': 'age',
    'noneconomic__crime': 'crime',
    'noneconomic__ethnicity': 'ethnicity',
    'noneconomic__family': 'family',
    'noneconomic__gender_sexuality': 'gender/sexuality',
    'noneconomic__health': 'health',
    'noneconomic__nationality': 'nationality',
    'noneconomic__place_location': 'place/location',
    'noneconomic__religion': 'religion',
    'noneconomic__shared_values_mentalities': 'shared values/mentalities',
}


label_cols = list(attribute_category_names_map.keys())
econ_attrs = [l for l in label_cols if l.startswith("economic__")]
nonecon_attrs = [l for l in label_cols if l.startswith("noneconomic__")]

econ_attr_names = [v for k, v in attribute_category_names_map.items() if k.startswith('economic__')]
nonecon_attr_names = [v for k, v in attribute_category_names_map.items() if k.startswith('noneconomic__')]
