import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

// services
import { ApiService } from './api.service';

// models
import { Availability } from '../_models/availability';

@Injectable({
  providedIn: 'root'
})
export class CalendarService {

  constructor() { }
}
