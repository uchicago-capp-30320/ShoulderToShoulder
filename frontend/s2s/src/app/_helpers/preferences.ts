// Description: Contains the preferences for the user to select from.
import { labelValueInt, labelValueIntArray } from "./constants";

export var timeCategories: string[] = [
    "Early morning (5-8a)",
    "Morning (9a-12p)",
    "Afternoon (1-4p)",
    "Evening (5-8p)",
    "Night (9p-12a)",
    "Late night (1-4a)",
    'Unavailable'
];

export var availableTimeCategoriesMap: labelValueIntArray[] = [
    {label: timeCategories[0], value: [5, 6, 7, 8]},
    {label: timeCategories[1], value: [9, 10, 11, 12]},
    {label: timeCategories[2], value: [13, 14, 15, 16]},
    {label: timeCategories[3], value: [17, 18, 19, 20]},
    {label: timeCategories[4], value: [21, 22, 23, 24]},
    {label: timeCategories[5], value: [1, 2, 3, 4]},
    {label: timeCategories[6], value: [0]}
];

// Convert to an object indexed by label
export const timeCategoryMap: { [label: string]: number[] } = availableTimeCategoriesMap.reduce((map, obj) => {
    map[obj.label] = obj.value;
    return map;
}, {} as { [label: string]: number[] });

export var availableTimes: labelValueInt[] = [
    {label: 'Unavailable', value: 0},
    {label: '1:00 AM', value: 1},
    {label: '2:00 AM', value: 2},
    {label: '3:00 AM', value: 3},
    {label: '4:00 AM', value: 4},
    {label: '5:00 AM', value: 5},
    {label: '6:00 AM', value: 6},
    {label: '7:00 AM', value: 7},
    {label: '8:00 AM', value: 8},
    {label: '9:00 AM', value: 9},
    {label: '10:00 AM', value: 10},
    {label: '11:00 AM', value: 11},
    {label: '12:00 PM', value: 12},
    {label: '1:00 PM', value: 13},
    {label: '2:00 PM', value: 14},
    {label: '3:00 PM', value: 15},
    {label: '4:00 PM', value: 16},
    {label: '5:00 PM', value: 17},
    {label: '6:00 PM', value: 18},
    {label: '7:00 PM', value: 19},
    {label: '8:00 PM', value: 20},
    {label: '9:00 PM', value: 21},
    {label: '10:00 PM', value: 22},
    {label: '11:00 PM', value: 23},
    {label: '12:00 AM', value: 24},
];

export var days = [
    'Monday',
    'Tuesday',
    'Wednesday',
    'Thursday',
    'Friday',
    'Saturday',
    'Sunday'
];

export var formControlFieldMap: { [index: string]: string } = {
    // Preferences
    'zipCode': 'Zip Code',
    'city': 'City',
    'state': 'State',
    'addressLine1': 'Address Line 1',
    'mostInterestedHobbies': 'Most Interested Hobbies',
    'leastInterestedHobbies': 'Least Interested Hobbies',
    'groupSizes': 'Group Sizes',
    'eventFrequency': 'Event Frequency',
    'eventNotifications': 'Event Notifications',
    'distances': 'Distances',

    // Demographics
    'groupSimilarity': 'Group Similarity',
    'groupSimilarityAttrs': 'Group Similarity Attributes',
    'ageRange': 'Age Range',
    'race': 'Race',
    'raceDesc': 'Race Description',
    'gender': 'Gender',
    'genderDesc': 'Gender Description',
    'sexualOrientation': 'Sexual Orientation',
    'sexualOrientationDesc': 'Sexual Orientation Description',
    'religiousAffiliation': 'Religious Affiliation',
    'religiousAffiliationDesc': 'Religious Affiliation Description',
    'politicalLeaning': 'Political Leaning',
    'politicalLeaningDesc': 'Political Leaning Description',

    // Event Availability
    'mondayTimes': 'Monday Times',
    'tuesdayTimes': 'Tuesday Times',
    'wednesdayTimes': 'Wednesday Times',
    'thursdayTimes': 'Thursday Times',
    'fridayTimes': 'Friday Times',
    'saturdayTimes': 'Saturday Times',
    'sundayTimes': 'Sunday Times',

    // Scenarios
    'scenario1': 'Scenario 1',
    'scenario1Choice': 'Scenario 1 Choice',
    'scenario2': 'Scenario 2',
    'scenario2Choice': 'Scenario 2 Choice',
    'scenario3': 'Scenario 3',
    'scenario3Choice': 'Scenario 3 Choice',
    'scenario4': 'Scenario 4',
    'scenario4Choice': 'Scenario 4 Choice',
    'scenario5': 'Scenario 5',
    'scenario5Choice': 'Scenario 5 Choice',
    'scenario6': 'Scenario 6',
    'scenario6Choice': 'Scenario 6 Choice',
    'scenario7': 'Scenario 7',
    'scenario7Choice': 'Scenario 7 Choice',
    'scenario8': 'Scenario 8',
    'scenario8Choice': 'Scenario 8 Choice'
};