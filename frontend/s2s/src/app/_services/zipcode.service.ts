import { Injectable } from '@angular/core';

// services
import { HttpClient } from '@angular/common/http';
import { ApiService } from './api.service';

/**
 * Service for the zipcode API. It calls the zipcodes api endpoint to query 
 * zipcode data.
 * 
 * Example:
 * ```
 * constructor(
 *  private zipcodeService: ZipcodeService,
 * ) { }
 * 
 * @see ApiService
 */
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

  /**
   * Gets the zipcode data from the zipcodes API.
   * 
   * @param zipcode The zipcode to query.
   * @returns The zipcode data.
   */
  getZipcode(zipcode: string) {
    return this.http.get(
      `${this.apiService.BASE_API_URL}/${this.enpoint}/?zip_code=${zipcode}`,
      this.httpOptions,);
  }
}
