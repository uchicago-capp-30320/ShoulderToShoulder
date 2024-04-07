import { Injectable } from '@angular/core';
import { FormBuilder, FormGroup, Validators, FormControl } from '@angular/forms';

// contants and helpers
import { 
  ageRanges, 
  races, 
  religiousAffiliations, 
  genders, 
  sexualOrientations 
} from '../_helpers/demographics';

@Injectable({
  providedIn: 'root'
})
export class UserService {
  public demographicsForm: FormGroup = this.fb.group({
    ageRange: new FormControl(''),
    race: new FormControl(''),
    raceDesc: new FormControl(''),
    gender: new FormControl(''),
    genderDesc: new FormControl(''),
    sexualOrientation: new FormControl(''),
    sexualOrientationDesc: new FormControl(''),
    religiousAffiliation: new FormControl(''),
    religiousAffiliationDesc: new FormControl(''),
  });
  ageRanges = ageRanges;
  races = races;
  religiousAffiliations = religiousAffiliations;
  genders = genders;
  sexualOrientations = sexualOrientations;

  constructor(private fb: FormBuilder) { }
}
