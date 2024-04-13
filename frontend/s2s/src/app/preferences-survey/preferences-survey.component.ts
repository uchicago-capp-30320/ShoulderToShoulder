import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';

// services
import { UserService } from '../_services/user.service';  
import { PreferencesService } from '../_services/preferences.service';

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
 */
@Component({
  selector: 'app-preferences-survey',
  templateUrl: './preferences-survey.component.html',
  styleUrl: './preferences-survey.component.css'
})
export class PreferencesSurveyComponent implements OnInit {
  hobbies!: string[];
  leastInterestedHobbies!: string[];
  mostInterestedHobbies!: string[];
  groupSizes = groupSizes;
  groupSimilarity = groupSimilarity;
  groupSimilarityAttrs = groupSimilarityAttrs;
  eventFrequency = eventFrequency;
  eventNotifications = eventNotifications;
  states = states;
  distances = distances;

  zipCodeApiUrl = "http://production.shippingapis.com/ShippingAPI.dll"
  zipCodeApi = "CityStateLookup"
  zipCodeApiKeyFilepath = "assets/api_keys/usps.txt"
  zipCodeXMLBase: string = "";
  zipCodeApiKey: string | null = null;

  constructor(
    public userService: UserService,
    private http: HttpClient,
    private preferencesService: PreferencesService
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
    this.hobbies = this.preferencesService.preferencesHobbies.map(hobby => hobby.name);
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
      this.zipCodeXMLBase = `<CityStateLookupRequest USERID="${data}">`
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
    let zipCodeXML = this.zipCodeXMLBase + `<ZipCode ID="0"><Zip5>${zipCode}</Zip5></ZipCode></CityStateLookupRequest>`
    let requestUrl = this.zipCodeApiUrl + `?API=${this.zipCodeApi}&XML=${zipCodeXML}`
    console.log(requestUrl)

    // sets city and state based on the response from the USPS API
    // TODO - uncomment once I have access to the API
    // this.http.get(requestUrl, { responseType: 'text' }).subscribe(data => {

    //   // USPS response is XML - need to parse
    //   try {
    //     const newData = this.parseXml(data);
    //     this.userService.preferencesForm.get('city')?.setValue(newData.city);
    //     this.userService.preferencesForm.get('state')?.setValue(newData.state);
    //   } catch (error) {
    //     console.error(error);
    //   }
    // });
  }

  /**
   * Parses the XML response from the USPS API.
   * 
   * @param xmlStr - string data to parse
   * @returns  - city and state data
   */
  private parseXml(xmlStr: string): any {
    // build parser
    const parser = new DOMParser();
    const xml = parser.parseFromString(xmlStr, "application/xml");
    const error = xml.getElementsByTagName("Error");

    // Handle any errors in the XML response
    if (error.length) {
      const description = error[0].getElementsByTagName("Description")[0].textContent;
      throw new Error((description as string));
    } 

    // Successfully parse the XML and extract city and state data
    else {
      const city = xml.getElementsByTagName("City")[0].textContent;
      const state = xml.getElementsByTagName("State")[0].textContent;
      return { 'city': city, 'state': state };
    }
  }

  /**
   * Submits the preferences form.
   */
  onSubmit() {
    console.log(this.userService.preferencesForm.value);
  }

}
