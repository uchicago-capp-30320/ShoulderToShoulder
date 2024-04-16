import { TestBed } from '@angular/core/testing';

import { UserService } from './user.service';

describe('UserService', () => {
  let service: UserService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(UserService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });

  it('should initialize the demographic form correctly', () => {
    const form = service.demographicsForm;
    expect(form).toBeTruthy();
    expect(form.get('ageRange')).toBeTruthy();
    expect(form.get('race')).toBeTruthy();
    expect(form.get('raceDesc')).toBeTruthy();
    expect(form.get('gender')).toBeTruthy();
    expect(form.get('genderDesc')).toBeTruthy();
    expect(form.get('sexualOrientation')).toBeTruthy();
    expect(form.get('sexualOrientationDesc')).toBeTruthy();
    expect(form.get('religiousAffiliation')).toBeTruthy();
    expect(form.get('religiousAffiliationDesc')).toBeTruthy();
    expect(form.get('politicalLeaning')).toBeTruthy();
    expect(form.get('politicalLeaningDesc')).toBeTruthy();
  });

  it('should initialize the preferences form correctly', () => {
    const form = service.preferencesForm;
    expect(form).toBeTruthy();
    expect(form.get('zipCode')).toBeTruthy();
    expect(form.get('city')).toBeTruthy();
    expect(form.get('state')).toBeTruthy();
    expect(form.get('addressLine1')).toBeTruthy();
    expect(form.get('mostInterestedHobbies')).toBeTruthy();
    expect(form.get('leastInterestedHobbies')).toBeTruthy();
    expect(form.get('groupSizes')).toBeTruthy();
    expect(form.get('groupSimilarity')).toBeTruthy();
    expect(form.get('groupSimilarityAttrs')).toBeTruthy();
    expect(form.get('eventFrequency')).toBeTruthy();
    expect(form.get('eventNotifications')).toBeTruthy();
    expect(form.get('distances')).toBeTruthy();
  });

  it('should be invalid when input is invalid - preferences form, zipcode', () => { 
    const form = service.preferencesForm;
    form.get('zipCode')?.setValue('1234');
    expect(form.get('zipCode')?.valid).toBeFalsy();
  });

  it('should be invalid when input is invalid - preferences form, mostInterestedHobbies', () => {
    const form = service.preferencesForm;
    form.get('mostInterestedHobbies')?.setValue([]);
    expect(form.get('mostInterestedHobbies')?.valid).toBeFalsy();
  });

  it('should be invalid when input is invalid - preferences form, groupSizes', () => {
    const form = service.preferencesForm;
    form.get('groupSizes')?.setValue([]);
    expect(form.get('groupSizes')?.valid).toBeFalsy();
  });

  it('should be invalid when input is invalid - preferences form, groupSimilarity', () => {
    const form = service.preferencesForm;
    form.get('groupSimilarity')?.setValue('');
    expect(form.get('groupSimilarity')?.valid).toBeFalsy();
  });

  it('should be invalid when input is invalid - preferences form, groupSimilarityAttrs', () => {
    const form = service.preferencesForm;
    form.get('groupSimilarityAttrs')?.setValue([]);
    expect(form.get('groupSimilarityAttrs')?.valid).toBeFalsy();
  });

  it('should be invalid when input is invalid - preferences form, eventNotifications', () => {
    const form = service.preferencesForm;
    form.get('eventNotifications')?.setValue('');
    expect(form.get('eventNotifications')?.valid).toBeFalsy();
  });

  it('should be invalid when input is invalid - preferences form, distances', () => {
    const form = service.preferencesForm;
    form.get('distances')?.setValue('');
    expect(form.get('distances')?.valid).toBeFalsy();
  });

  it('should initialize the event availability form correctly', () => {
    const form = service.eventAvailabilityForm;
    expect(form).toBeTruthy();
    expect(form.get('mondayTimes')).toBeTruthy();
    expect(form.get('tuesdayTimes')).toBeTruthy();
    expect(form.get('wednesdayTimes')).toBeTruthy();
    expect(form.get('thursdayTimes')).toBeTruthy();
    expect(form.get('fridayTimes')).toBeTruthy();
    expect(form.get('saturdayTimes')).toBeTruthy();
    expect(form.get('sundayTimes')).toBeTruthy();
  });

  it('should initialize the scenarios form correctly', () => {
    const form = service.scenariosForm;
    expect(form).toBeTruthy();
    expect(form.get('scenario1')).toBeTruthy();
    expect(form.get('scenario1Scenario')).toBeTruthy();
    expect(form.get('scenario2')).toBeTruthy();
    expect(form.get('scenario2Scenario')).toBeTruthy();
    expect(form.get('scenario3')).toBeTruthy();
    expect(form.get('scenario3Scenario')).toBeTruthy();
    expect(form.get('scenario4')).toBeTruthy();
    expect(form.get('scenario4Scenario')).toBeTruthy();
    expect(form.get('scenario5')).toBeTruthy();
    expect(form.get('scenario5Scenario')).toBeTruthy();
    expect(form.get('scenario6')).toBeTruthy();
    expect(form.get('scenario6Scenario')).toBeTruthy();
    expect(form.get('scenario7')).toBeTruthy();
    expect(form.get('scenario7Scenario')).toBeTruthy();
    expect(form.get('scenario8')).toBeTruthy();
    expect(form.get('scenario8Scenario')).toBeTruthy();
  });

  it('should be invalid when input is invalid - scenarios form, scenario1', () => {
    const form = service.scenariosForm;
    form.get('scenario1')?.setValue(undefined);
    expect(form.get('scenario1')?.valid).toBeFalsy();
  });

  it('should be invalid when input is invalid - scenarios form, scenario1Scenario', () => {
    const form = service.scenariosForm;
    form.get('scenario1Scenario')?.setValue(undefined);
    expect(form.get('scenario1Scenario')?.valid).toBeFalsy();
  });

  it('should be invalid when input is invalid - scenarios form, scenario2', () => {
    const form = service.scenariosForm;
    form.get('scenario2')?.setValue(undefined);
    expect(form.get('scenario2')?.valid).toBeFalsy();
  });

  it('should be invalid when input is invalid - scenarios form, scenario2Scenario', () => {
    const form = service.scenariosForm;
    form.get('scenario2Scenario')?.setValue(undefined);
    expect(form.get('scenario2Scenario')?.valid).toBeFalsy();
  });

  it('should be invalid when input is invalid - scenarios form, scenario3', () => {
    const form = service.scenariosForm;
    form.get('scenario3')?.setValue(undefined);
    expect(form.get('scenario3')?.valid).toBeFalsy();
  });

  it('should be invalid when input is invalid - scenarios form, scenario3Scenario', () => {
    const form = service.scenariosForm;
    form.get('scenario3Scenario')?.setValue(undefined);
    expect(form.get('scenario3Scenario')?.valid).toBeFalsy();
  });

  it('should be invalid when input is invalid - scenarios form, scenario4', () => {
    const form = service.scenariosForm;
    form.get('scenario4')?.setValue(undefined);
    expect(form.get('scenario4')?.valid).toBeFalsy();
  });

  it('should be invalid when input is invalid - scenarios form, scenario4Scenario', () => {
    const form = service.scenariosForm;
    form.get('scenario4Scenario')?.setValue(undefined);
    expect(form.get('scenario4Scenario')?.valid).toBeFalsy();
  });

  it('should be invalid when input is invalid - scenarios form, scenario5', () => {
    const form = service.scenariosForm;
    form.get('scenario5')?.setValue(undefined);
    expect(form.get('scenario5')?.valid).toBeFalsy();
  });

  it('should be invalid when input is invalid - scenarios form, scenario5Scenario', () => {
    const form = service.scenariosForm;
    form.get('scenario5Scenario')?.setValue(undefined);
    expect(form.get('scenario5Scenario')?.valid).toBeFalsy();
  });

  it('should be invalid when input is invalid - scenarios form, scenario6', () => {
    const form = service.scenariosForm;
    form.get('scenario6')?.setValue(undefined);
    expect(form.get('scenario6')?.valid).toBeFalsy();
  });

  it('should be invalid when input is invalid - scenarios form, scenario6Scenario', () => {
    const form = service.scenariosForm;
    form.get('scenario6Scenario')?.setValue(undefined);
    expect(form.get('scenario6Scenario')?.valid).toBeFalsy();
  });

  it('should be invalid when input is invalid - scenarios form, scenario7', () => {
    const form = service.scenariosForm;
    form.get('scenario7')?.setValue(undefined);
    expect(form.get('scenario7')?.valid).toBeFalsy();
  });

  it('should be invalid when input is invalid - scenarios form, scenario7Scenario', () => {
    const form = service.scenariosForm;
    form.get('scenario7Scenario')?.setValue(undefined);
    expect(form.get('scenario7Scenario')?.valid).toBeFalsy();
  });

  it('should be invalid when input is invalid - scenarios form, scenario8', () => {
    const form = service.scenariosForm;
    form.get('scenario8')?.setValue(undefined);
    expect(form.get('scenario8')?.valid).toBeFalsy();
  });

  it('should be invalid when input is invalid - scenarios form, scenario8Scenario', () => {
    const form = service.scenariosForm;
    form.get('scenario8Scenario')?.setValue(undefined);
    expect(form.get('scenario8Scenario')?.valid).toBeFalsy();
  });

});
