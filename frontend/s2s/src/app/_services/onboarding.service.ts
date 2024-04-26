import { Injectable } from '@angular/core';
import { FormBuilder, FormGroup, Validators, FormControl } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { catchError } from 'rxjs/operators';
import { EMPTY } from 'rxjs';

// services
import { CalendarService } from './calendar.service';
import { ApiService } from './api.service';
import { AuthService } from './auth.service';

// helpers
import { NumberRegx } from '../_helpers/patterns';
import { ScenarioObj } from '../_models/scenarios';
import { Scenario } from '../_helpers/scenario';
import { Onboarding } from '../_models/onboarding';
import { User } from '../_models/user';

@Injectable({
  providedIn: 'root'
})
export class OnboardingService {
  onboardingEndpoint = this.apiService.BASE_API_URL + '/onboarding/';
  scenariosEndpoint = this.apiService.BASE_API_URL + '/scenarios/';

  // onboarding forms
  public demographicsForm: FormGroup = this.fb.group({
    groupSimilarity: new FormControl('', Validators.required),
    groupSimilarityAttrs: new FormControl([], Validators.required),
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
    scenario1Choice: new FormControl(undefined, Validators.required),
    scenario2: new FormControl(undefined, Validators.required),
    scenario2Choice: new FormControl(undefined, Validators.required),
    scenario3: new FormControl(undefined, Validators.required),
    scenario3Choice: new FormControl(undefined, Validators.required),
    scenario4: new FormControl(undefined, Validators.required),
    scenario4Choice: new FormControl(undefined, Validators.required),
    scenario5: new FormControl(undefined, Validators.required),
    scenario5Choice: new FormControl(undefined, Validators.required),
    scenario6: new FormControl(undefined, Validators.required),
    scenario6Choice: new FormControl(undefined, Validators.required),
    scenario7: new FormControl(undefined, Validators.required),
    scenario7Choice: new FormControl(undefined, Validators.required),
    scenario8: new FormControl(undefined, Validators.required),
    scenario8Choice: new FormControl(undefined, Validators.required),
  });
  onboarded: boolean = false;

  constructor(
    private fb: FormBuilder,
    public calendarService: CalendarService,
    public authService: AuthService,
    private http: HttpClient,
    private apiService: ApiService
  ) { }

  submitOnboardingForms(): void {
    this.authService.user.subscribe(user => {
      console.log(user)
      this.submitOnboarding(user);
      this.submitScenarios(user);
      this.submitAvailabilityForm();
    });
  }

  submitOnboarding(user: User): void {
    // collect data
    let onboarding: Onboarding = {
      user_id: user.id,
      onboarded: true,
      num_participants: this.preferencesForm.get('groupSizes')?.value,
      distance: this.preferencesForm.get('distances')?.value,
      similarity_to_group: this.demographicsForm.get('groupSimilarity')?.value,
      similarity_metrics: this.demographicsForm.get('groupSimilarityAttrs')?.value,
      gender: this.demographicsForm.get('gender')?.value,
      gender_description: this.demographicsForm.get('genderDesc')?.value,
      race: this.demographicsForm.get('race')?.value,
      race_description: this.demographicsForm.get('raceDesc')?.value,
      age: this.demographicsForm.get('ageRange')?.value,
      sexual_orientation: this.demographicsForm.get('sexualOrientation')?.value,
      sexual_orientation_description: this.demographicsForm.get('sexualOrientationDesc')?.value,
      religion: this.demographicsForm.get('religiousAffiliation')?.value,
      religion_description: this.demographicsForm.get('religiousAffiliationDesc')?.value,
      political_leaning: this.demographicsForm.get('politicalLeaning')?.value,
      political_description: this.demographicsForm.get('politicalLeaningDesc')?.value,
    }

    // send onboarding data to the backend
    this.http.post(this.onboardingEndpoint, onboarding).pipe(
      catchError(error => {
        console.error('Error submitting onboarding:', error);
        return EMPTY;
      })
    ).subscribe(() => {
      console.log('Onboarding submitted successfully!');
    });
  }

  submitScenarios(user: User): void {
    // collect data
    let scenarioObjs: ScenarioObj[] = [];

    for (let i = 1; i <= 8; i++) {
      let scenario: Scenario = this.scenariosForm.get(`scenario${i}`)?.value;
      let scenarioObj: ScenarioObj = scenario.scenarioObj;
      let choice = this.scenariosForm.get(`scenario${i}Choice`)?.value;

      scenarioObj.user_id = user.id;
      scenarioObj.prefers_event1 = choice == 1 ? true : false;
      scenarioObj.prefers_event2 = !scenarioObj.prefers_event1;

      scenarioObjs.push(scenarioObj);

      // send scenario data to the backend
      this.http.post(this.scenariosEndpoint, scenarioObj).pipe(
        catchError(error => {
          console.error('Error submitting scenarios:', error);
          return EMPTY;
        })
      ).subscribe(() => {
        console.log('Scenarios submitted successfully!');
      });
    }
  }

  submitAvailabilityForm(): void {
    // send availability data to the backend
    this.calendarService.updateAvailability();
  }

}
