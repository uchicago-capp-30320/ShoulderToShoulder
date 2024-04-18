// Description: Contains the preferences for the user to select from.
import { labelValueInt, labelValueIntArray } from "./constants";

/**
 * An interface for representing a hobby that a user can select.
 */
export interface Hobby {
    name: string;
    scenarioForm: string;
    maxParticipants: number;
    type: string;

}

export var hobbies: Hobby[] = [
    { name: 'Arcade bar', scenarioForm: 'an arcade bar', maxParticipants: 10, type: 'GAMING' },
    { name: 'Art Museums', scenarioForm: 'an art museum', maxParticipants: 10, type: 'ARTS AND CULTURE' },
    { name: 'Attending Book Signing', scenarioForm: 'a book signing', maxParticipants: 5, type: 'LITERATURE' },
    { name: 'Attending Neighborhood Parade', scenarioForm: 'a neighborhood parade', maxParticipants: 20, type: 'COMMUNITY EVENTS' },
    { name: 'Biking', scenarioForm: 'go on a bike ride', maxParticipants: 10, type: 'SPORT/EXERCISE' },
    { name: 'Board Games', scenarioForm: 'play board games', maxParticipants: 4, type: 'GAMING' },
    { name: 'Book Club', scenarioForm: 'a book club', maxParticipants: 10, type: 'LITERATURE' },
    { name: 'Bouldering', scenarioForm: 'go bouldering', maxParticipants: 5, type: 'SPORT/EXERCISE' },
    { name: 'Bowling', scenarioForm: 'go bowling', maxParticipants: 8, type: 'SPORT/EXERCISE' },
    { name: 'Chess', scenarioForm: 'play chess', maxParticipants: 1, type: 'GAMING' },
    { name: 'City tour', scenarioForm: 'go on a city tour', maxParticipants: 5, type: 'HISTORY AND LEARNING' },
    { name: 'Collaging', scenarioForm: 'a collaging circle', maxParticipants: 5, type: 'CRAFTING' },
    { name: 'Comedy Clubs', scenarioForm: 'go to a comedy club', maxParticipants: 5, type: 'ARTS AND CULTURE' },
    { name: 'Concert', scenarioForm: 'go to a concert', maxParticipants: 20, type: 'ARTS AND CULTURE' },
    { name: 'Exploring a Nearby Town', scenarioForm: 'explore a nearby town', maxParticipants: 5, type: 'TRAVEL' },
    { name: 'Exploring breweries', scenarioForm: 'explore breweries', maxParticipants: 5, type: 'FOOD AND DRINK' },
    { name: 'Gardening', scenarioForm: 'go gardening', maxParticipants: 5, type: 'OUTDOORS' },
    { name: 'Exploring coffee shops', scenarioForm: 'explore coffee shops', maxParticipants: 5, type: 'FOOD AND DRINK' },
    { name: 'Going to the beach', scenarioForm: 'go to the beach', maxParticipants: 10, type: 'OUTDOORS' },
    { name: 'Hiking', scenarioForm: 'go on a hike', maxParticipants: 10, type: 'SPORT/EXERCISE' },
    { name: 'History Museums', scenarioForm: 'a history museum', maxParticipants: 10, type: 'HISTORY AND LEARNING' },
    { name: 'Jazz Concerts', scenarioForm: 'go to a jazz concert', maxParticipants: 10, type: 'ARTS AND CULTURE' },
    { name: 'Jigsaw Puzzles', scenarioForm: 'do a jigsaw puzzle', maxParticipants: 5, type: 'GAMING' },
    { name: 'Knitting', scenarioForm: 'a knitting circle', maxParticipants: 5, type: 'CRAFTING' },
    { name: 'Live amateur sporting event', scenarioForm: 'an amateur sporting event', maxParticipants: 10, type: 'SPORT/EXERCISE' },
    { name: 'Picnic', scenarioForm: 'go on a picnic', maxParticipants: 5, type: 'OUTDOORS' },
    { name: 'Playing baseball/softball', scenarioForm: 'play softball/baseball', maxParticipants: 10, type: 'SPORT/EXERCISE' },
    { name: 'Playing basketball', scenarioForm: 'play basketball', maxParticipants: 10, type: 'SPORT/EXERCISE' },
    { name: 'Playing Soccer', scenarioForm: 'play soccer', maxParticipants: 11, type: 'SPORT/EXERCISE' },
    { name: 'Playing Tennis', scenarioForm: 'play tennis', maxParticipants: 4, type: 'SPORT/EXERCISE' },
    { name: 'Playing volleyball', scenarioForm: 'play volleyball', maxParticipants: 10, type: 'SPORT/EXERCISE' },
    { name: 'Poetry Writing', scenarioForm: 'attend a poetry workshop', maxParticipants: 5, type: 'LITERATURE' },
    { name: 'Professional Sporting Events', scenarioForm: 'a professional sporting event', maxParticipants: 20, type: 'SPORT/EXERCISE' },
    { name: 'Quilting', scenarioForm: 'a quilting circle', maxParticipants: 5, type: 'CRAFTING' },
    { name: 'Role Playing Video Games', scenarioForm: 'play a role-playing video game', maxParticipants: 10, type: 'GAMING' },
    { name: 'Rollerskating', scenarioForm: 'go rollerskating', maxParticipants: 5, type: 'SPORT/EXERCISE' },
    { name: 'Running (club)', scenarioForm: 'a running club', maxParticipants: 10, type: 'SPORT/EXERCISE' },
    { name: 'Skateboarding', scenarioForm: 'go skateboarding', maxParticipants: 5, type: 'SPORT/EXERCISE' },
    { name: 'Swimming', scenarioForm: 'go swimming', maxParticipants: 10, type: 'SPORT/EXERCISE' },
    { name: 'Taking Cooking Classes', scenarioForm: 'take a cooking class', maxParticipants: 5, type: 'COOKING/BAKING' },
    { name: 'Trying New Restaurants', scenarioForm: 'try a new restaurant', maxParticipants: 5, type: 'COOKING/BAKING' },
    { name: 'Volunteer Work', scenarioForm: 'volunteer', maxParticipants: 10, type: 'COMMUNITY EVENTS' },
    { name: 'Volunteering at an animal shelter', scenarioForm: 'volunteer at an animal shelter', maxParticipants: 10, type: 'COMMUNITY EVENTS' },
    { name: 'Walking (club)', scenarioForm: 'a walking club', maxParticipants: 10, type: 'SPORT/EXERCISE' },
    { name: 'Watching Ballet', scenarioForm: 'see a ballet performance', maxParticipants: 5, type: 'ARTS AND CULTURE' },
    { name: 'Watching Musicals', scenarioForm: 'see a musical', maxParticipants: 5, type: 'ARTS AND CULTURE' },
    { name: 'Watching Opera', scenarioForm: 'see an opera performance', maxParticipants: 5, type: 'ARTS AND CULTURE' },
    { name: 'Watching Plays', scenarioForm: 'see a play', maxParticipants: 5, type: 'ARTS AND CULTURE' }
];


export var groupSizes = [
    '1-5',
    '6-10',
    '10-15',
    '15+',
    'No preference'
];

export var groupSimilarity = [
    'Completely dissimilar',
    'Moderately dissimilar',
    'Neutral',
    'Moderately similar',
    'Completely similar',
    'No preference'
];

export var groupSimilarityAttrs = [
    "Age range",
    "Gender",
    "Political Leaning",
    "Race or Ethnicity",
    "Religious Affiliation",
    "Sexual Orientation",
    "No preference"
];

export var eventFrequency = [
    'Twice a week',
    'Once a week',
    'Once every two weeks',
    'Once a month',
    'Once every three months',
];

export var eventNotifications = [
    'Email Only',
    'Email and Text',
    'Text Only',
];

export var distances = [
    'Within 1 mile',
    'Within 5 miles',
    'Within 10 miles',
    'Within 15 miles',
    'Within 20 miles',
    'Within 30 miles',
    'Within 40 miles',
    'Within 50 miles',
    'No preference'
]

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

export var minMileage = 0;
export var maxMileage1 = 10;
export var maxMileage2 = 50;

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