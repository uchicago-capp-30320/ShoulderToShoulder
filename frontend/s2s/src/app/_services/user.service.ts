import { Injectable } from '@angular/core';
import { FormBuilder, FormGroup, Validators, FormControl } from '@angular/forms';

@Injectable({
  providedIn: 'root'
})
export class UserService {
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
    hobbies: new FormControl('', Validators.required),
    groupSizes: new FormControl('', Validators.required),
    groupSimilarity: new FormControl('', Validators.required),
    groupSimilarityAttrs: new FormControl('', Validators.required),
    eventFrequency: new FormControl(''),
    eventNotifications: new FormControl('', Validators.required),
    eventNotificationFrequency: new FormControl('', Validators.required),
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


  constructor(
    private fb: FormBuilder
    ) { }
}
