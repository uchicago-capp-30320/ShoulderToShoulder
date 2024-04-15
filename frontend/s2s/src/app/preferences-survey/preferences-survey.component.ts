import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';

// services
import { UserService } from '../_services/user.service';  
import { HobbyService } from '../_services/hobbies.service';

// helpers
import { 
  groupSizes, 
  groupSimilarity, 
  groupSimilarityAttrs, 
  eventFrequency, 
  eventNotifications,
  distances
} from '../_helpers/preferences';

import { states } from '../_helpers/location';

/**
 * Component for the preferences survey form. Also contains the logic for
 * extracting the zip code data from the form and sending a request to the USPS 
 * API.
 * 
 * Example:
 * ```
 * <app-preferences-survey></app-preferences-survey>
 * ```
 * 
 * @see UserService
 * @see HobbyService
 */
@Component({
  selector: 'app-preferences-survey',
  templateUrl: './preferences-survey.component.html',
  styleUrl: './preferences-survey.component.css'
})
export class PreferencesSurveyComponent implements OnInit {
  // hobby information
  hobbies!: string[];
  leastInterestedHobbies!: string[];
  mostInterestedHobbies!: string[];

  // group information
  groupSizes = groupSizes;
  groupSimilarity = groupSimilarity;
  groupSimilarityAttrs = groupSimilarityAttrs;

  // event information
  eventFrequency = eventFrequency;
  eventNotifications = eventNotifications;

  // location information
  states = states;
  distances = distances;
  zipCodeApiUrl = "https://api.zipcodestack.com/v1/search?country=us"
  zipCodeApiKeyFilepath = "assets/api_keys/zipcodestack.txt"
  zipCodeApiKey: string | null = null;
  zipcodeInvalid: boolean = false;

  constructor(
    public userService: UserService,
    private http: HttpClient,
    private HobbyService: HobbyService
  ) {
    this.getHobbyArray();
  }

  ngOnInit() {
    this.getZipCodeApiKey()
  }

  /**
   * Extracts an array of hobby names from the Hobby[] list. Sets this.hobbies
   * to the hobby array.
   */
  getHobbyArray() {
    this.hobbies = this.HobbyService.preferencesHobbies.map(hobby => hobby.name);
    this.leastInterestedHobbies = this.hobbies;
    this.mostInterestedHobbies = this.hobbies;
  }

  /**
   * Extracts an array of hobbies that are in the most interested hobbies array.
   */
  getMostInterestedHobbiesArray() {
    this.mostInterestedHobbies = this.hobbies.filter(hobby => !this.userService.preferencesForm.get('leastInterestedHobbies')?.value.includes(hobby));
  }

  /**
   * Extracts an array of hobbies that are not in the most interested hobbies array.
   */
  getLeastInterestedHobbiesArray() {
    this.leastInterestedHobbies = this.hobbies.filter(hobby => !this.userService.preferencesForm.get('mostInterestedHobbies')?.value.includes(hobby));
  }

  /**
   * Gets the USPS API key from the .txt file.
   * 
   */
  getZipCodeApiKey() {
    this.http.get(this.zipCodeApiKeyFilepath).subscribe(data => {
      this.zipCodeApiKey = (data as string);
    })
  }

  /**
   * Extracts the zip code data from the preferences form and sends a request to 
   * the USPS API to get the city and state data.
   * 
   * @returns null if the zip code is null.
   * 
   */
  getZipCodeData() {
    // extracts the zipcode from the form
    console.log("Getting zip code data")
    let zipCode = this.userService.preferencesForm.get('zipCode')?.value
    if (zipCode == null) {
      console.log("Zip code is null")
      return
    }

    // builds the API request URL
    let requestUrl = this.zipCodeApiUrl + `&codes=${zipCode}&apikey=${this.zipCodeApiKey}`
    console.log(requestUrl)

    // sets city and state based on the response from the USPS API
    this.http.get(requestUrl, { responseType: 'json' }).subscribe(data => {

      // ZipCodeStack response is JSON - need to parse
      try {
        let results = (data as any).results

        // determine if the zipcode exists
        if (results == null || results.length == 0) {
          this.zipcodeInvalid = true;
          return
        }

        // extracts city and state from the zipcode
        let result = (data as any).results[zipCode][0]
        console.log(result)
        let city = result.city
        let state = {label: result.state, value: result.state_code}

        console.log(city, state)
        this.userService.preferencesForm.get('city')?.setValue(city)
        this.userService.preferencesForm.get('state')?.setValue(state)

        console.log(this.userService.preferencesForm.value)
      } catch (error) {
        console.error(error);
      }
    });
  }

  /**
   * Submits the preferences form.
   */
  onSubmit() {
    console.log(this.userService.preferencesForm.value);
  }

}
