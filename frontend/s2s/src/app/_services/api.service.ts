import { Injectable } from '@angular/core';
import { environment } from '../../environments/environment';

/**
 * Service for managing API calls. Interacts with the environment file to get
 * the base API URL and the application token.
 *
 * @example
 * ```
 * this.http.get(`${this.apiService.BASE_API_URL}/endpoint/`);
 * ```
 */
@Injectable({
  providedIn: 'root'
})
export class ApiService {
  public BASE_API_URL = environment.api;
  public BASE_S2S_URL = environment.s2s;
  public production = environment.production;
  public appToken = environment.appToken;

  constructor() {
   }
}
