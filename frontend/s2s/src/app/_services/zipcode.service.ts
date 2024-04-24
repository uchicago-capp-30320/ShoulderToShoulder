import { Injectable } from '@angular/core';

// services
import { HttpClient } from '@angular/common/http';
import { ApiService } from './api.service';

@Injectable({
  providedIn: 'root'
})
export class ZipcodeService {
  enpoint = 'zipcodes';
  httpOptions = {
    headers: {
      'Content-Type': 'application/json',
    },
  }

  constructor(
    private http: HttpClient,
    private apiService: ApiService,
  ) { }

  getZipcode(zipcode: string) {
    return this.http.get(
      `${this.apiService.BASE_API_URL}/${this.enpoint}/?zip_code=${zipcode}`,
      this.httpOptions,);
  }
}
