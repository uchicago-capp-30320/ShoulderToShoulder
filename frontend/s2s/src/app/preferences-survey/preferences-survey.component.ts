import { Component, OnInit } from '@angular/core';
import { UserService } from '../_services/user.service';  
import { HttpClient } from '@angular/common/http';

import { 
  hobbies, 
  groupSizes, 
  groupSimilarity, 
  groupSimilarityAttrs, 
  eventFrequency, 
  eventNotificationFrequency, 
  eventNotifications 
} from '../_helpers/preferences';

@Component({
  selector: 'app-preferences-survey',
  templateUrl: './preferences-survey.component.html',
  styleUrl: './preferences-survey.component.css'
})
export class PreferencesSurveyComponent implements OnInit {
  hobbies = hobbies;
  groupSizes = groupSizes;
  groupSimilarity = groupSimilarity;
  groupSimilarityAttrs = groupSimilarityAttrs;
  eventFrequency = eventFrequency;
  eventNotificationFrequency = eventNotificationFrequency;
  eventNotifications = eventNotifications;

  zipCodeApiUrl = "https://api.usps.com/addresses/v3/city-state"
  zipCodeApiKeyFilepath = "assets/api_keys/zipcode.txt"
  zipCodeApiHeaders: any;
  zipCodeApiKey: string | null = null;

  constructor(
    public userService: UserService,
    private http: HttpClient
  ) {}

  ngOnInit() {
    this.getZipCodeApiKey()
  }

  getZipCodeApiKey() {
    this.http.get(this.zipCodeApiKeyFilepath).subscribe(data => {
      this.zipCodeApiKey = (data as string);
      this.zipCodeApiHeaders = {
        "apikey": `${data}`,
        "Accept": "application/json",
      }

      console.log(this.zipCodeApiHeaders)
    })
  }

  getZipCodeData() {
    console.log("Getting zip code data")
  }

  /**
   * Submits the preferences form.
   */
  onSubmit() {
    console.log(this.userService.preferencesForm.value);
  }

}
