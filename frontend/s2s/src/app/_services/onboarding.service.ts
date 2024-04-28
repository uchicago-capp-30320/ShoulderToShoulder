import { Injectable } from '@angular/core';
import { FormBuilder, FormGroup, Validators, FormControl } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { catchError } from 'rxjs/operators';
import { EMPTY, race } from 'rxjs';

// services
import { CalendarService } from './calendar.service';
import { ApiService } from './api.service';
import { AuthService } from './auth.service';

// helpers
import { NumberRegx } from '../_helpers/patterns';
import { ScenarioObj } from '../_models/scenarios';
import { Scenario } from '../_helpers/scenario';
import { Onboarding, OnboardingResp } from '../_models/onboarding';
import { User } from '../_models/user';
import { getState } from '../_helpers/utils';

@Injectable({
  providedIn: 'root'
})
export class OnboardingService {
  onboardingEndpoint = this.apiService.BASE_API_URL + '/onboarding/';
  onboardingUpdateEndpoint = this.apiService.BASE_API_URL + '/onboarding/update/';
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
  ) { 
    this.authService.user.subscribe(user => {
      if (user && user.id > -1) {
        this.fetchOnboarding();
      }
    });
  }

  /**
   * Fetches any existing onboarding data for the user. This function allows
   * a user to start the onboarding process, exit it, and return later without
   * losing their progress.
   * 
   * @returns 
   */
  fetchOnboarding(): void {
    let user = this.authService.userValue;
    this.http.get<OnboardingResp>(`${this.onboardingEndpoint}?user_id=${user.id}`).pipe(
      catchError(error => {
        console.error('Error fetching onboarding:', error);
        return EMPTY;
      })
    ).subscribe(onboardingResp => {
      if (onboardingResp) {
        let onboarding = onboardingResp.results[0];
        console.log('Onboarding fetched successfully!')
        console.log(onboarding);
        this.onboarded = onboarding.onboarded;
        this.setDemographicsForm(onboarding);
        this.setPreferencesForm(onboarding);
        this.calendarService.loadAllCalendar();
      }
    });
  }

  /**
   * Sets the demographics form with the user's existing onboarding data.
   * 
   * @param onboarding 
   */
  setDemographicsForm(onboarding: Onboarding): void {
    this.demographicsForm.setValue({
      groupSimilarity: onboarding.similarity_to_group,
      groupSimilarityAttrs: onboarding.similarity_metrics,
      ageRange: onboarding.age,
      race: onboarding.race,
      raceDesc: onboarding.race_description,
      gender: onboarding.gender,
      genderDesc: onboarding.gender_description,
      sexualOrientation: onboarding.sexual_orientation,
      sexualOrientationDesc: onboarding.sexual_orientation_description,
      religiousAffiliation: onboarding.religion,
      religiousAffiliationDesc: onboarding.religion_description,
      politicalLeaning: onboarding.political_leaning,
      politicalLeaningDesc: onboarding.political_description,
    });
  }

  /**
   * Sets the preferences form with the user's existing onboarding data.
   * 
   * @param onboarding 
   */
  setPreferencesForm(onboarding: Onboarding): void {
    this.preferencesForm.setValue({
      zipCode: onboarding.zip_code,
      city: onboarding.city,
      state: getState(onboarding.state),
      addressLine1: onboarding.address_line1,
      mostInterestedHobbies: onboarding.most_interested_hobbies,
      leastInterestedHobbies: onboarding.least_interested_hobbies,
      groupSizes: onboarding.num_participants,
      eventFrequency: onboarding.event_frequency,
      eventNotifications: onboarding.event_notification,
      distances: onboarding.distance,
    });
  }

  /**
   * Exist onboarding by sending current data to the backend and
   * signing user out.
   */
  exitOnboarding(submit: boolean = false): void {
    let user = this.authService.userValue;

    if (submit) {
      this.submitOnboarding(user);
      this.submitScenarios(user);
      this.submitAvailabilityForm();
    }

    this.authService.logout();
  }

  /**
   * Submits the onboarding forms to the backend.
   */
  submitOnboardingForms(): void {
    let user = this.authService.userValue;
    this.submitOnboarding(user);
    this.submitScenarios(user);
    this.submitAvailabilityForm();
  }

  submitOnboarding(user: User, onboarded: boolean = true): void {
    // collect data
    let onboarding: Onboarding = {
      user_id: user.id,
      onboarded: onboarded,

      // preferences form
      most_interested_hobbies: this.getStringToListChar("mostInterestedHobbies", this.preferencesForm),
      least_interested_hobbies: this.getStringToListChar("leastInterestedHobbies", this.preferencesForm),
      num_participants: this.getStringToListChar("groupSizes", this.preferencesForm), 
      distance: this.preferencesForm.get('distances')?.value,
      zip_code: this.preferencesForm.get('zipCode')?.value,
      city: this.preferencesForm.get('city')?.value,
      state: (this.preferencesForm.get('state')?.value as {label: string, value: string}).value,
      address_line1: this.preferencesForm.get('addressLine1')?.value,
      event_frequency: this.preferencesForm.get('eventFrequency')?.value,
      event_notification: this.preferencesForm.get('eventNotifications')?.value,

      // demographics form
      similarity_to_group: this.demographicsForm.get('groupSimilarity')?.value,
      similarity_metrics: this.getStringToListChar("groupSimilarityAttrs", this.demographicsForm), 
      gender: this.getStringToListChar("gender", this.demographicsForm), 
      gender_description: this.demographicsForm.get('genderDesc')?.value,
      race: this.getStringToListChar("race", this.demographicsForm), 
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
    this.http.post(this.onboardingUpdateEndpoint, onboarding).pipe(
      catchError(error => {
        console.error('Error submitting onboarding:', error);
        return EMPTY;
      })
    ).subscribe(() => {
      console.log('Onboarding submitted successfully!');
    });
  }

  getStringToListChar(controlName: string, form: FormGroup): string | string[] {
    let char = form.get(controlName)?.value;
    if (!char || char.length == 0) {
      return "";
    }
    return char; 
  }

  submitScenarios(user: User): void {
    // collect data
    let scenarioObjs: ScenarioObj[] = [];

    for (let i = 1; i <= 8; i++) {
      let scenario: Scenario = this.scenariosForm.get(`scenario${i}`)?.value;
      if (scenario) {
        let scenarioObj: ScenarioObj = scenario.scenarioObj;
        let choice = this.scenariosForm.get(`scenario${i}Choice`)?.value;

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
  }

  submitAvailabilityForm(): void {
    // send availability data to the backend
    this.calendarService.updateAvailability();
  }
}
