import { Injectable } from '@angular/core';
import { environment } from '../../environments/environment';

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