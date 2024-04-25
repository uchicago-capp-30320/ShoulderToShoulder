import { Component, OnInit } from '@angular/core';

// services
import { HobbyService } from '../_services/hobbies.service';
import { OnboardingService } from '../_services/onboarding.service';
import { ZipcodeService } from '../_services/zipcode.service';
import { ChoicesService } from '../_services/choices.service';

// helpers
import { states } from '../_helpers/location';

@Component({
  selector: 'app-preferences-survey',
  templateUrl: './preferences-survey.component.html',
  styleUrls: ['./preferences-survey.component.css']
})
export class PreferencesSurveyComponent implements OnInit {
  hobbies: string[] = [];
  leastInterestedHobbies: string[] = [];
  mostInterestedHobbies: string[] = [];
  states = states;
  zipcodeInvalid = false;
  choices: { [index: string]: any[]; } = {};

  constructor(
    public onboardingService: OnboardingService, 
    private hobbyService: HobbyService, 
    private zipCodeService: ZipcodeService,
    private choicesService: ChoicesService,
  ) {}

  ngOnInit() {
    this.getHobbyArray();
    this.getChoices();
  }

  getChoices() {
    this.choicesService.choices.subscribe(choices => {
      this.choices = choices;
    });
  }

  getHobbyArray() {
    this.hobbyService.hobbies.subscribe(hobbies => {
      this.hobbies = hobbies.map(hobby => hobby.name);
      this.leastInterestedHobbies = this.hobbies.slice();
      this.mostInterestedHobbies = this.hobbies.slice();
    });
  }

  /**
   * Extracts an array of hobbies that are in the most interested hobbies array.
   */
  getMostInterestedHobbiesArray() {
    this.mostInterestedHobbies = this.hobbies.filter(hobby => !this.onboardingService.preferencesForm.get('leastInterestedHobbies')?.value.includes(hobby));
  }

  /**
   * Extracts an array of hobbies that are not in the most interested hobbies array.
   */
  getLeastInterestedHobbiesArray() {
    this.leastInterestedHobbies = this.hobbies.filter(hobby => !this.onboardingService.preferencesForm.get('mostInterestedHobbies')?.value.includes(hobby));
  }

  /**
   * Extracts the zip code data from the preferences form and sends a request to 
   * the zipcode API endpoint to get the city and state data.
   * 
   * @returns null if the zip code is null.
   * 
   */
  getZipCodeData() {
    // extracts the zipcode from the form
    let zipCode = this.onboardingService.preferencesForm.get('zipCode')?.value
    if (zipCode == null) {
      return
    }

    // sets city and state based on the response from the API
    this.zipCodeService.getZipcode(zipCode).subscribe(data => {

      // ZipCodeStack response is JSON - need to parse
      try {
        let results = (data as any).results

        // determine if the zipcode exists
        if (results == null || results.length == 0) {
          this.zipcodeInvalid = true;
          return
        }

        this.zipcodeInvalid = false;

        // extracts city and state from the zipcode
        let result = (data as any).results[zipCode][0]
        let city = result.city
        let state = {label: result.state, value: result.state_code}

        this.onboardingService.preferencesForm.get('city')?.setValue(city)
        this.onboardingService.preferencesForm.get('state')?.setValue(state)

      } catch (error) {
        console.error(error);
      }
    });
  }

  /**
   * Submits the preferences form.
   */
  onSubmit() {
    console.log(this.onboardingService.preferencesForm.value);
  }

}
