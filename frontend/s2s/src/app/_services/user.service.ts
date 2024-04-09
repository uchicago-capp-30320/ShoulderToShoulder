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
    hobbies: new FormControl('', Validators.required),
    groupSizes: new FormControl('', Validators.required),
    groupSimilarity: new FormControl('', Validators.required),
    groupSimilarityAttrs: new FormControl('', Validators.required),
    eventFrequency: new FormControl(''),
    eventNotifications: new FormControl('', Validators.required),
    eventNotificationFrequency: new FormControl('', Validators.required),
  });

  public eventAvailabilityForm: FormGroup = this.fb.group({
    mondayTimes: new FormControl([0]),
    tuesdayTimes: new FormControl([0]),
    wednesdayTimes: new FormControl([0]),
    thursdayTimes: new FormControl([0]),
    fridayTimes: new FormControl([0]),
    saturdayTimes: new FormControl([0]),
    sundayTimes: new FormControl([0]),
  });

  // TODO - conver these values from bools to 0-1
  public scenariosForm: FormGroup = this.fb.group({
    scenario1: new FormControl(''),
    scenario2: new FormControl(''),
    scenario3: new FormControl(''),
    scenario4: new FormControl(''),
    scenario5: new FormControl(''),
    scenario6: new FormControl(''),
    scenario7: new FormControl(''),
    scenario8: new FormControl(''),
    scenario9: new FormControl(''),
    scenario10: new FormControl(''),
  });

  onboarded: boolean = false;
  user: User | null = null;

  constructor(
    private fb: FormBuilder
    ) { }
}
