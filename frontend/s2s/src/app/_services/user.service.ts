import { Injectable } from '@angular/core';
import { FormBuilder, FormGroup, Validators, FormControl } from '@angular/forms';
import { NumberRegx } from '../_helpers/patterns';
import { User } from '../_helpers/userInfo';

@Injectable({
  providedIn: 'root'
})
export class UserService {

  // onboarding forms
  public demographicsForm: FormGroup = this.fb.group({
    ageRange: new FormControl(''),
    race: new FormControl(''),
    raceDesc: new FormControl(''),
    gender: new FormControl(''),
    genderDesc: new FormControl(''),
    sexualOrientation: new FormControl(''),
    sexualOrientationDesc: new FormControl(''),
    religiousAffiliation: new FormControl(''),
    religiousAffiliationDesc: new FormControl(''),
    politicalLeaning: new FormControl(''),
    politicalLeaningDesc: new FormControl(''),
  });

  public preferencesForm: FormGroup = this.fb.group({
    zipCode: new FormControl('', [
      Validators.minLength(5),
      Validators.maxLength(5),
      Validators.required,
      Validators.pattern(NumberRegx)
    ]),
    city: new FormControl(''),
    state: new FormControl(''),
    addressLine1: new FormControl(''),
    mostInterestedHobbies: new FormControl([], Validators.required),
    leastInterestedHobbies: new FormControl([]),
    groupSizes: new FormControl([], Validators.required),
    groupSimilarity: new FormControl('', Validators.required),
    groupSimilarityAttrs: new FormControl([], Validators.required),
    eventFrequency: new FormControl(''),
    eventNotifications: new FormControl('', Validators.required),
    distances: new FormControl('', Validators.required),
  });

  public eventAvailabilityForm: FormGroup = this.fb.group({
    mondayTimes: new FormControl([]),
    tuesdayTimes: new FormControl([]),
    wednesdayTimes: new FormControl([]),
    thursdayTimes: new FormControl([]),
    fridayTimes: new FormControl([]),
    saturdayTimes: new FormControl([]),
    sundayTimes: new FormControl([]),
  });

  public scenariosForm: FormGroup = this.fb.group({
    scenario1: new FormControl(undefined, Validators.required),
    scenario1Scenario: new FormControl(undefined, Validators.required),
    scenario2: new FormControl(undefined, Validators.required),
    scenario2Scenario: new FormControl(undefined, Validators.required),
    scenario3: new FormControl(undefined, Validators.required),
    scenario3Scenario: new FormControl(undefined, Validators.required),
    scenario4: new FormControl(undefined, Validators.required),
    scenario4Scenario: new FormControl(undefined, Validators.required),
    scenario5: new FormControl(undefined, Validators.required),
    scenario5Scenario: new FormControl(undefined, Validators.required),
    scenario6: new FormControl(undefined, Validators.required),
    scenario6Scenario: new FormControl(undefined, Validators.required),
    scenario7: new FormControl(undefined, Validators.required),
    scenario7Scenario: new FormControl(undefined, Validators.required),
    scenario8: new FormControl(undefined, Validators.required),
    scenario8Scenario: new FormControl(undefined, Validators.required),
  });

  onboarded: boolean = false;
  user: User | null = null;

  constructor(
    private fb: FormBuilder
    ) { }
}
