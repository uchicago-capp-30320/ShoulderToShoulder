import { TestBed } from '@angular/core/testing';

import { ZipcodeService } from './zipcode.service';

describe('ZipcodeService', () => {
  let service: ZipcodeService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(ZipcodeService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
